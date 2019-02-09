import dropbox

dbx = dropbox.Dropbox('JFJlsVJ7QeAAAAAAAAADgAQ3tzcDQVYhr1Oyxyj1_nnPTZSDhdb5tk7uKehjCfRW')

def upload (file, name):
    dbx.files_delete(name)
    dbx.files_upload(file, name, mode=dropbox.files.WriteMode.overwrite)

def load (download_path, path):
    dbx.files_download_to_file(download_path, path, rev=None)
