import json 
from indra_nxml_extraction import id_lookup
from download_pmc_s3 import download_pmc_s3 
import os
from indra_nxml_extraction import get_xml, get_xml_from_file, extract_text
from lxml import etree
from bs4 import BeautifulSoup


def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data    


def write_json_file(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def convert_pmcid_from_pmid():
    for pmid in pmid_list:
        pmcid = id_lookup(pmid)
        pmcid_list.append(pmcid)
    return pmcid_list


def read_and_extract_xml_data(fileName):
    with open(fileName, 'r') as f:
        data = f.read()
    xml_data = BeautifulSoup(data, "xml")
    text = extract_text(xml_data)
    return text


pmid_list, no_pmid_list, pmcid_list, no_xml, pmcid_no_xml = [], [], [], [], []
allStudy = load_json_file('pubmed/cBioportal_study.json')           # all study =411

# FIRSR STEP
# make a list for studies have pmid and a list without pmid
for study in allStudy:
    pmid_value = study.get('pmid')
    if pmid_value:  
        pmid_list.append(pmid_value)                                #pmid = 351
    else:  # some study have no pmid
        no_pmid_list.append(study)                                  # no pmid = 60

print(len(pmid_list))
print(len(no_pmid_list))
# write_json_file('no_pmid.json',no_pmid_list)


# SECOND STEP
# # convert pmcid from pmid, then write a file for pmcid            #pmcid = 351
# pmcid_list = convert_pmcid_from_pmid()
# write_json_file('pmcid_list.json', pmcid_list)
pmcid_data = load_json_file('pubmed/pmcid_list.json')


# THIRD STEP
# download xml file using pmcids, and extract content from xml papers
i = 0
j = 0
k = 0
for study in pmcid_data:                                            # 351
    pmcid = study.get('pmcid')
    try: 
        if pmcid:  
            file_path = f'loaded_pmc/{pmcid}.txt'
            if not os.path.exists(file_path):
                # download xml file using aws S3
                try: 
                    download_pmc_s3(pmcid)
                    fileName = str('pmc/' + pmcid + '.xml')
                    try:
                        # read and extract xml data
                        text = read_and_extract_xml_data(fileName)
                        # write a txt file for extract xml content
                        with open('loaded_pmc/' + pmcid + '.txt', 'w') as f:
                            f.write(str(text))
                    except Exception as e:                                  #29
                        j += 1
                        print(f"Error read and extract pmcid {pmcid}: {e}")

                except Exception as e: 
                    k += 0
                    print(f"can't be downloaded {pmcid}: {e}")

        else:  # pmcid = null                                           #40
            no_xml.append(study)

    except Exception as e:  # study has pmcid, but no xml form
        i += 1
        print(f"Error downloading pmcid {pmcid}: {e}")
        pmcid_no_xml.append(study)
        print("This study" + pmcid + " has pmcid, but no xml paper")

print("failed to download file number: " + str(i))
print("failed to extract and load file number: " + str(j))
print("total cannot be downloaded: " + str(k))
write_json_file('no_xml.json', no_xml)  
write_json_file('pmcid_no_xml.json', pmcid_no_xml)
