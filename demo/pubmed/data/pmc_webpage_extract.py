import requests, lxml
from bs4 import BeautifulSoup

headers = {
    'User-agent':
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6419906/'
url_cBio = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4048021/'
soup = BeautifulSoup(requests.get(url_cBio, headers=headers).text, 'lxml')

body = soup.body
with open('pubmed_webfile_body.txt', 'w') as file:
    file.write(str(body))

for tag in body.select('script'):
    tag.decompose()
for tag in body.select('style'):
    tag.decompose()

text = body.get_text(separator='\n').strip()
# with open('pubmed_webfile.txt', 'w') as file:
#     file.write(text)
print(text)