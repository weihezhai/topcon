import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = 'sl.u.AF7Drccv6pYQmyLpjXCeZ3XCeWFwOQq7-5Z-E3RvqVoGCmY9iT8W5Wdbd5as2xUO9X6KHmuH_tJ1ZWtsUILlVc2Jy0_bl7nY5E3p_GMbsVhhRTVJR_URqOI1yNvkB5BJJSmF7PE-ks4gEZCicw5j3cwWwUyLAuSAmSl3C89MUPQuvn557TrlfHGhoTv0IbLSm5Q04uG7BKy2ch3LOFaIbn7KtmuQIpfKfa7eEEvH6z0F-mXUs94k7OgyrRMi7Ttq7hffH5UuMjcfLPL7OfG6qgZYghj_5V2F1It0pJ4yrXqwWnjIQWIFfIisLDXFNNXnJXU0HJbhfvPDJ-wWjPmTq-hR2aaY5RbVeGInjb82IQdkRcd9268mOU-UwKCx8kBx7_gjaMzI0Oy9kRIQDSH4qxyYKft2L92hgWjXYUDZzfiUvqON2oJVvvJoK3nbCt9pmramemiq9UHGJO4l9dgFz27Mp4XSCYRRviv1s9zS6CSEEsSvP9tpKUzpksVX6t7vPUSJTg_gEJqEeqsRlf_Dk9VIxTen_QjalqLzpWm6wgP-MZyvBKV7chIa5SgYJDrOhqWH9uMFpMDfKtKv2pYxRR5XwhdzPzNircpUOD1iKEtzo7bawPcc-GxLV6LT-q_vRnaqTgOOq8EnEcCmhU92CwNZvRjgRWp8O4IlCln5SHbvSQW0AIF1ljQa9CRFRWgWwTp-HWCTzigf3M4ivg73v8D8_LfDFg1pAD8v2euvM8uAtE9K_2d953lNhdFKQWnTOdxYQdUlt3DbGS9Uxx2Wpap4x8CcebqI26MSDLIbooYHsK5k9n-VhqqZtnJFPlw1MiDmn0DlF7QDEXKJPSB0CcatF_nmuSjNNwJ2-lWElop6bRQ6d5sqiMzCY-Fs17tR68Hv2OfPFGJoZ6r5oxv9QxffEkphukyKVc-xW1X51c1RJlj3mjvi7TIPIrO3h355-bqPo_lpeVkmEtK1iZn0r81JNvhcSrxY06y2xz1O8dRNer0YXp74w8RqfdhKU90N8XL3yrp55dh0l2feJGXrkbCXwdoRgfWiYkwsPQ0Px1UU0VGSHQxn32ROzxXKhzGxGttt6aEKaHWbiAQ-Nqru0IJQnCd6xi69C-IJEEeA18zK7tHYbWmS663azLbx4FwOWtO2AKgOAzZwFBB-ag2ob88_9feve5NAmohut0RM9AUkbMxBVrDUFM_xdALRM3ll4SU4ymIZ3HgN6P2ZbAncMGAlcGGdlkAWwh9E0USqwjvglCB8UaKUzwCSARxBT3EM7M4'
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