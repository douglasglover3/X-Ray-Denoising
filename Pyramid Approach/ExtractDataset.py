import os
import glob
import urllib.request
import tarfile

def extract_data():
    # If the data is not already extracted manually, extract it
    extract_loc = 'Data/'
    if (not os.path.exists(extract_loc)):
        os.mkdir(extract_loc)
    if (not os.path.exists('Noisy/')):
        os.mkdir('Noisy/')
    if (not os.path.exists('Output/')):
        os.mkdir('Output/')
    # URLs for the zip files (This code comes from the batch_download_zips.py code provided in the NIH dataset)
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    tar_gz_files = glob.glob(os.path.join('', '*.tar.gz'))
    images = glob.glob(os.path.join(os.path.join(extract_loc, 'Images/'), '*.png'))
    if not tar_gz_files and not images:
        for idx, link in enumerate(links):
            fn = 'images_%02d.tar.gz' % (idx + 1)
            print('downloading ' + fn + '...')
            if (not os.path.exists(fn)):
                urllib.request.urlretrieve(link, fn)  # download the zip file
        print("Download complete. Please check the checksums")

    if tar_gz_files:
        for file in tar_gz_files:
            with tarfile.open(file, "r:gz") as tar:
                tar.extractall(path='Data/')
                print(f"Extracted contents to {os.path.join(os.getcwd(), extract_loc)}")
            os.remove(file)