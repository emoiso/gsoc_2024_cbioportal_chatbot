import json 
from data.indra_nxml_extraction import id_lookup
from data.download_pmc_s3 import download_pmc_s3 
import os
from data.indra_nxml_extraction import get_xml, get_xml_from_file, extract_text
from lxml import etree
from bs4 import BeautifulSoup


def read_and_extract_xml_data(fileName):
    with open(fileName, 'r') as f:
        data = f.read()
    xml_data = BeautifulSoup(data, "xml")
    text = extract_text(xml_data)
    return text


def add_new_paper(pmid):
    try: 
        pmcid = id_lookup(pmid).get("pmcid")
        try: 
            download_pmc_s3(pmcid)
            fileName = str('pmc/' + pmcid + '.xml')
            try:
                # read and extract xml data
                text = read_and_extract_xml_data(fileName)
                print(text)
                # write a txt file for extract xml content
                with open(f'loaded_pmc/{pmcid}.txt', 'w') as f:
                    f.write(str(text))

            except Exception as e:                                 
                print(f"Error read and extract pmcid {pmcid}: {e}")     
        except Exception as e:
            print(f"can't be downloaded {pmcid}: {e}")

    except Exception as e: 
        print(f"This {pmid} has no pmcid: {e}")


add_new_paper("36517593")
