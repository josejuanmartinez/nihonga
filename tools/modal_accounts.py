"""Select credentials explicitly. The old account is restricted to archive reads."""
import os


def credential(name):
    if os.name=='nt':
        import winreg
        for hive,path in ((winreg.HKEY_CURRENT_USER,'Environment'),
                          (winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')):
            try:
                with winreg.OpenKey(hive,path) as key:
                    value=winreg.QueryValueEx(key,name)[0]
                if value:return value
            except FileNotFoundError:pass
    return os.environ.get(name)


def set_modal_account(account='current', *, purpose):
    if account not in ('current','old'):raise ValueError(account)
    if account=='old' and purpose!='download':
        raise RuntimeError('The old Modal account is download-only. Computation and uploads must use the new account.')
    suffix='_OLD' if account=='old' else ''
    token_id=credential('MODAL_TOKEN_ID'+suffix)
    token_secret=credential('MODAL_TOKEN_SECRET'+suffix)
    if not token_id or not token_secret:raise RuntimeError('Missing '+account+' account credentials')
    if purpose!='download' and token_id==credential('MODAL_TOKEN_ID_OLD'):
        raise RuntimeError('Refusing to compute or upload using the old account token')
    os.environ['MODAL_TOKEN_ID']=token_id
    os.environ['MODAL_TOKEN_SECRET']=token_secret
    return account
