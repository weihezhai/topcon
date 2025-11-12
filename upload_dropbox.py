import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGHmQBfNhP4Kt_8QAD-pyCckkVahoVOQGCa5n3AJAMz75AdnTY3rjkScXULa3Ey1S47l4vaHEgy0uHWUPyuOWtC17crxCAQi5ZriCAfIjgNc3x52qru4Om1d3TLGkNhLTVFRD0WtjxXpJnxtCiM_aBwJkYUXueKqspIrHG6pivNgBkiuga3KsqU3kGTzXuHpM3ihktOCMCTa6UIfsdXJV7dBGvBQtxff4A_3mOBNt4_QYwjSnj7WoIkP--3GpsYCp4_d_qFtztMaGOXJubYTdyH7i0I2itXhdK3sxsGs2QIPhKqmCHy_Hn9W3zmWKtGumzhT0r8ge_EXohXSk9Vw_rJyN66COJGilgxf43mim0Wiaygyanedt7H13UqNjl-hCGfxPjYpFT3UYVFqU5pZkLPlgvWzApQusCmUEIOwWtLGuoHuPMQmWb8Xn_evlUBsfiHiKv9y4Ts9J8rgxZtwz4VvvEa7kaB110OkiEjQEWF66FmfPXCFC_PRSF-L2jHpioxvGbk0jMVyoEW9OYbmZeIDsdaQvJvI3kJ8Kiuq4ue9c-puOMWi-fNRqwnhhmHc7E3VLh4hn8B85o8Kb6vCCNK7R5qM4mVSTv8hcIBHhBAuRBhDW5RNgfExlZK53uD0IoiF4jJlNHh9PnMXB8NTnHmb25K-Sr77SjwYstE8-qkit7MDhLFvMpNEf634pXUIRmirG_8gIXL6dUL0dYwE-7M23Gs-7yfQhwaIl2-gnh7Mfi0lSNZunT4o6kcNbVMhXwNZNvu5F6Ig2yKNOVwnk44s_d0YUVcCa17hAp_wb4FB8rLe95313o9eqw-uIoJfqurpDnfaolcOU8ceu_vqzlzzCHnUZrfLruQyE_btSUpASlqIqC9_q0CR6EdcuYVLu3T61jqIqidfB7ZZdecus75S-rFH7JUFuHTu16fNcv1IUKTKvCLloiwKLjNTk9RQiormI-RNnBqnKtiB19_YQ1pWOoCQHuCQOs06DCh6oLeoagolEI4qm33FjNnkzjy2ENTXQuvTFpl5sOG-mBpQUQWEprIDYytgxyne8bVfoeAhFrO10vHoGHCFDxhhvRAN0hzbvBjsXDvSQyWzvJN5JbDN_cXpWWb9GAPbJuNt8dDsgCzRsKICzpsUb0DdZiGSxUZVrqLJeQApHM2NugYQgu2UF7QibE1Xgh6wUd-wAxE9Kfv0fGq0Y_-emxabECqBLnmdCilNshsG1W-75kgDHRUXEZ2vsjg64nBVJaioai22QcvWRtZ-mY9SGm6qzYVjlKw"
CHUNK_SIZE = 150 * 1024 * 1024  # 150MB - A good default, adjustable based on network stability

def upload_large_file(local_path, dropbox_path):
    """
    Uploads a large file to Dropbox using chunked uploads with a static token.
    """
    if not ACCESS_TOKEN:
        print("Error: Please set the DROPBOX_ACCESS_TOKEN environment variable.")
        sys.exit(1)

    try:
        # Instantiate the Dropbox object with the static access token
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
        dbx.users_get_current_account() # Test the connection
        print("Successfully connected to Dropbox.")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        sys.exit(1)

    file_size = os.path.getsize(local_path)
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")

    with open(local_path, 'rb') as f:
        try:
            if file_size <= CHUNK_SIZE:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
                print(f"Successfully uploaded {local_path} to {dropbox_path}")
            else:
                upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                           offset=f.tell())
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode('overwrite'))

                print(f"Started upload session: {cursor.session_id}")

                while f.tell() < file_size:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                         cursor,
                                                         commit)
                        print("\nFile upload complete.")
                    else:
                        try:
                            dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                            cursor.offset = f.tell()
                            # Progress indicator
                            progress = (f.tell() / file_size) * 100
                            sys.stdout.write(f"\rProgress: {progress:.2f}%")
                            sys.stdout.flush()
                        except dropbox.exceptions.ApiError as e:
                            if e.error.is_incorrect_offset():
                                print(f"\nOffset error. Server has {e.error.get_incorrect_offset().correct_offset} bytes. Retrying from that offset.")
                                f.seek(e.error.get_incorrect_offset().correct_offset)
                                cursor.offset = e.error.get_incorrect_offset().correct_offset
                            else:
                                raise

        except Exception as e:
            print(f"\nAn error occurred during upload: {e}")
            sys.exit(1)

def upload_folder(local_folder, dropbox_folder):
    """
    Recursively uploads all files in a folder to Dropbox.
    """
    if not ACCESS_TOKEN:
        print("Error: Please set the DROPBOX_ACCESS_TOKEN environment variable.")
        sys.exit(1)

    try:
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
        dbx.users_get_current_account()
        print("Successfully connected to Dropbox.")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        sys.exit(1)

    total_files = sum([len(files) for _, _, files in os.walk(local_folder)])
    uploaded_files = 0
    
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_file_path, local_folder)
            dropbox_file_path = os.path.join(dropbox_folder, relative_path).replace(os.sep, '/')
            
            uploaded_files += 1
            print(f"\n[{uploaded_files}/{total_files}] Uploading: {relative_path}")
            upload_large_file(local_file_path, dropbox_file_path)

    print(f"\nFolder upload complete. Uploaded {uploaded_files} files.")

if __name__ == "__main__":
    '''
    example usage:
    python upload_dropbox.py /data/scratch/mpx602/topcon-1/neurips2024.tar.gz /Paper_pred/neurips2024.tar.gz
    python upload_dropbox.py /mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/mineru_data /Paper_pred/mineru_data
    '''
    if len(sys.argv) != 3:
        print("Usage: python upload_script.py <local_path> <dropbox_destination_path>")
        print("       local_path can be a file or a folder")
        sys.exit(1)

    local_path = sys.argv[1]
    dropbox_dest = sys.argv[2]

    if not os.path.exists(local_path):
        print(f"Error: Local path '{local_path}' not found.")
        sys.exit(1)

    if os.path.isdir(local_path):
        print(f"Detected folder: {local_path}")
        upload_folder(local_path, dropbox_dest)
    else:
        print(f"Detected file: {local_path}")
        upload_large_file(local_path, dropbox_dest)