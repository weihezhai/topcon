import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGEL3xV_qrIp4yhg_DfBL1G9XLPvggmXVtGFTKDYnW_SgR2PdEyoCGBSmxg8vJoke5IBOdcqDwQyNcAX0FP1iT4vNo3cF51_D1h6SgWqeP3-2WANS1X8y1elG6oYbYK_7W-VnURsNDm7FKyn7xVJo0eercoSB-1kmICmliHECcaN7GUbjaZVbS5ZbtIk7_j2hsar6u0e1t6lGNn0GqtlHXJsCLQPTrD5XvFzxi-3OkRO7VbZ4C5DPUAaO13IWvp-AKNqCvAy8UZzVrpOZU6ZiM42rrq7O0aGmi4fjNPdTJzPVfSi1HuGOFgUHF-_HwkZMytvIYpnuTraoqvNNTlJ5i6QUNjwYrJgfi563bI_E1QnfDLGYYCJjVya5aXiWUFAO0Xbi3USg9vudXeGtuu9PS89b_Om03qzhHLHxgK61ip7inKGSJPpgbAPtPfdxdJrwiLBaxMoN84WTr6SNa7SjWf_Ww6dZoCQXGYesKohl4heM2UyK0YAFqfX2-XywaCDdaAbhC3Z6beD4rFHG1MKF9TU7FP_ntresrLqxONWCLC9oXjSkJG9RK1-pXn-j34b9yXzaMBIAADy6F8WgiP-DQwyVMtFwH8tkcWvmOXJ3E54cJcX5umM-Q6fuZG1Oz4Q0b5Q4hpHSpny6B9l7ixduNqpd3anZ01IBvDdYTBNsJN7C3qPw-RooirqgK_zB7ZTrHJrHLjA_BcBYcVA_3ZLLkYBrxisORrFAqdsqvGBi_i61tAZQlTxiwD6LhRgkS-Jo2voyZCUZDpsUS7pGYP6J_htqL5tL817Y_YjrRup_nGSMqDJgRplPNrqKF0xSU-_C4U_QWwOa6fd_88mEuFK-GmqF2aWm-n9szYDk9Krj_khaAPtBo-29k3JynT7eorlgFYby9Zk5eBHTHD4j0NqK7eHxnTBkrBFi1vcyFqWVjK_dv2bRUNXkZIsAv1QhYxf4vOaXvRGRXiCpRbmw3rlYI6wkq61K4bXpURSXKYwVkA7FGvV3WktJ0E8x2H6ZLcqrqZmzJc_Pv-TSJ8o7ZLFiNZsSDFECG_foNYgErWqAqixRfASRQPHRWdDULzZvwF8qTicUyF5SFON7cRriZkszJ0xu64oHI_CAGG8jhpEiDvDx6hv96TFXaYExMy1bQ6nJGLUIbDju2c-yvw1dkEfnKOLCfeFVyzjmw716usukk-rTsJf96J7-Q8Sf8wHuKphwB7_yYgPBQ_xFmyef-M6kq8xyXDuLgtwBqppmKMs7KRGXalhLk95sQD4gXRzzjGP6cE"
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
    python upload_dropbox.py /userfolder/neurips2025.tar.gz /Paper_pred/mineru_data/neurips2025.tar.gz
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