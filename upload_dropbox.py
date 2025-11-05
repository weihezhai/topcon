import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGHOxrreioW9stVQNmO5UU_wd5tt9Fmz3BEUqO5StEvCN7KAG6PHtuPdqaOXLCbrfuxqCD_WXmLZGC2bKyI4aqVYx8YDbdkGtYxjL3d-oX9UzhEF_rlZ-hbHxH-E12gEOcdxOrNrpK5i9wtdrhjd-bnYRKNVnGYcqgl4JtNtO6CdGAD_KA992xgKzgCTwHegoy1MqNvmHTy4-gTCsgcg0KwXLAtY12YwZm7jAAtqitQLKiIarrbpQMyZTeAryNbSSvIl4_1qocetOZmqoa9s5rcvYjeMOC7azvlgivpy6lEZz9Xz3E-W6DI7HwuzQpokHI15XBDYcTj1Abs2ONUTrgYYhzJkcE8th2gulqisdAWeiilLGIt_ubYeC_TuWv0IcagXFoXfI0HuqGlMZN8OCthaE5FWmXhUXMLnTeYPOr0GNhIynzIb6W0mmFCFosXmAfH-vNsHuYW0wDcOSu4_vUewUFPp6sEXlgBZkAYB2pWlbTR26_LNI7jXzLVy2tubW5TcSdunwSwIRbhHpsyX_piQ9NgNgSfKhFr83XAT6eLglFwirZEddti7VokTLlg50X3P2d9gmSS40-OjfSVhNmVa7oGPAq-NO8NUJnLqIYx7QjPdKqx5RVh4mDhXpkmR-Y1INDvo8pKIdVYWIIEzYhVX9NfDFDFMFViimsyXmbLqiis-_UifiOfVIsITPBZWqRLUP0f6CezQYUa7LMfpWn7JTU4YZKi7IXta3YBkx6nt7n3ZTzCm3lav9PQvgoRHxNewwTcdgPDm-4Coniz82UjWqdJqSBlW_aZcOh5EjEuI_lfobE-fjkw7Zay3466FFgIsToAnx4XRJtXaMs6ARYmLwPUr6zq95qGxVsJqROHiDAv7jCQU-_OATgAgS-e7y7t6bgmCq_Y0tlvmovY5CId19kD4LUntS-4b2OgHxDEhB1nLLLb3qhQMtS9nc78HRwuLhLNZCNRdH-q0zznBHNYcVZ-onF6QT63HLocgYk51zrcQB2cZnJ_XuYDC-KIlEeSMbIjWwnsVdPp_No6u5Jues-sN4vqV2bTCBqsv37Lk912i3nkwsWkCn9eDDfCFsPaV3BAbbQZzg-3Ol2DdVQQh_U3HCbKnCvePdHQW0EyqMSBRWVVJ8IX4jDVHRfVExh7FriDyiDDuzZhcPaCuCto34TVMvYz1YIl0GvFNt-sQaF2z2og3YqmN5oBFEGiV9f3FnWt__FoMCdQsnM2jt3DO6FgFMUT3UnA2vFgBB9vLgHYkDazc_HH2XPGredcwQ9I"
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