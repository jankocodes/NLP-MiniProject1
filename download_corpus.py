import os
import requests
import zipfile

def download_text8(url='http://mattmahoney.net/dc/text8.zip', filename='text8.zip'):
    if not os.path.exists(filename):
        print('Downloading Text8 dataset...')
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print('Download completed.')
    else:
        print('Text8 dataset already exists.')

def extract_first_n_words(zip_filename='text8.zip', output_filename='text8_20m.txt', n_words=2000000):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall()
    
    # Read the text8 file
    with open('text8', 'r') as f:
        text = f.read()
    
    # Split into wordss
    words = text.split()
    
    # Take the first n_words
    selected_words = words[:n_words]
    
    # Write to the output file
    with open(output_filename, 'w') as f:
        f.write(' '.join(selected_words))
    
    print(f'Saved first {n_words} words to {output_filename}')

if __name__ == '__main__':
    download_text8()
    extract_first_n_words()