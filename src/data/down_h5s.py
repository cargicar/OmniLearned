from dataloader import get_url, download_h5_files

url = f"https://portal.nersc.gov/cfs/m4567/jetclass/val/"

destination = f"/home/carlos/Rnet_local/datasets/jetclass"
download_h5_files(url, destination)