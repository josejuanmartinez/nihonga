"""Account isolation checks: never spend GPU or upload using archive credentials."""
import os
import unittest
from unittest.mock import patch

from tools import modal_accounts


class AccountIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tokens={'MODAL_TOKEN_ID':'new-id','MODAL_TOKEN_SECRET':'new-secret',
                     'MODAL_TOKEN_ID_OLD':'old-id','MODAL_TOKEN_SECRET_OLD':'old-secret'}

    def test_old_account_cannot_compute_or_upload(self):
        for purpose in ('compute','upload'):
            with self.subTest(purpose=purpose),self.assertRaises(RuntimeError):
                modal_accounts.set_modal_account('old',purpose=purpose)

    def test_old_downloads_remain_available(self):
        with patch.object(modal_accounts,'credential',self.tokens.get),patch.dict(os.environ,{},clear=True):
            modal_accounts.set_modal_account('old',purpose='download')
            self.assertEqual(os.environ['MODAL_TOKEN_ID'],'old-id')

    def test_compute_replaces_inherited_old_credentials_with_new(self):
        with patch.object(modal_accounts,'credential',self.tokens.get),patch.dict(os.environ,{'MODAL_TOKEN_ID':'old-id'},clear=True):
            modal_accounts.set_modal_account('current',purpose='compute')
            self.assertEqual(os.environ['MODAL_TOKEN_ID'],'new-id')
            self.assertEqual(os.environ['MODAL_TOKEN_SECRET'],'new-secret')

    def test_misconfigured_current_token_is_rejected(self):
        self.tokens['MODAL_TOKEN_ID']='old-id'
        with patch.object(modal_accounts,'credential',self.tokens.get),self.assertRaises(RuntimeError):
            modal_accounts.set_modal_account('current',purpose='compute')

    def test_missing_new_credentials_do_not_fall_back_to_old(self):
        del self.tokens['MODAL_TOKEN_ID']
        with patch.object(modal_accounts,'credential',self.tokens.get),self.assertRaises(RuntimeError):
            modal_accounts.set_modal_account('current',purpose='compute')


if __name__=='__main__':unittest.main()
