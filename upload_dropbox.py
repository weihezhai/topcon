import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGECh7jZFC369p6_YffrxI7ucoYLXH7qJZfUyneVkDKpjPZoxFThDgt29_OL9uGVY0xrRZsdvLeT4m6898WZWXYDSkBOP1-rO9kSYn2Z46We1-zIZTUlg6iRTmmz2y-sAo-70uZMitDPocWiucKIMG_aMkWrksk3x4b9BJbn3rwQkldxLZPfh_6DAodAE9KV3sn76QIKZcH7p_ABKbWkqFmoIOULqpclaiv3K6H77Twz3SFA_Ekl-AhYj6LJtxFXP0R8i-ecOxr0VMt4wF1J6qOtVVBbld5wU4zyTmjKKRcmMK0WK_iYDm2grdoRgLhZaNMeKZL2b55gjpkkS4uHuRoRjV6q8jKKXPkok6Wi8D6Rjy-uUUZtGIVb2NAGJ-dxdgD2iwn3udlNzVHhH2HErHA2C1i4TPql479LYzqnuyJ73GAwgeankPYwyA9ZMvI32mBMdTURLEuIN1ezoKSIFMavUeczbL8zaXpusuUWsvs5QlTUSyHnA3GXLNZSlOYrIbshV56osZ0hO_w3ZzHx0mR3_DqhNV3NHuIVVPPX-Pyssmd7trNzO-TEBlM5ty-6SYtKj1g8xnSrsM12LN_kNTb9mxrYcHvOv9HZuWkfUoCWiUT-NJfXpOqrRvcrl4U9ou-2MB-MPHSIg7RpP2U55LPVg3odNZL1Bivaj5tZFIgdrMFDGg0nBrp8H4WZl0_jBI4hCZVcxEr0wWoB256ij1SldVnddzYoj4yWqI-bxxCrGajxX49e5B__LH5h6-jqdG1blBDEkwLVp6cggA_0LOQT5a07MCPVLDQxm6GNnajdwnKi8C8IOwzAssSHNI0YnVlz1m2h2L-9R0HPNxXmv9DzsE4-mrkp5ca25fKgslW1Ur618LQQYQZ3c2dY8Y8ii2MHtpWBts641t0OJxt84JOKfzIHqNhLMZTfsbfaL4H-ZHaThj8GLizUunesttxXeuj8qqdPcKNg2dIsnLttlHUBhWGXLmShxLBQmxq0JwFAogCj2T9FZoMX1lOe82ObT1c2NMgsME7wSu4I_RZDvwI-qpZLhedG_irZg-7xP-yApW1gBwRkmE5fRk23mPDDENciCWeb6E0G6B0yt4fYHzjk8rypjPZlo1eQYyHEOMdFaFKr6qLr0jagTCCl_YqoD4elLrE2E6-f7pI3axNTw_yYArOHPvBGCpbTKxgpRNbZKzqy9JmsA4XrgSFBklxYF_EM_66oHRiiJY3rt474mDotUar-H-UY5UD7vM39PGN5aVx4Xx16FdhdkLfr2r-TjfU"
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