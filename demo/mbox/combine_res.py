import json


# this script combines messages from same google conversation
def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data 


def write_json_file(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


messages = load_json_file('mbox/mbox_meassage.json')
n = len(messages)
new_messages = [messages[-1]]

print(n)
subject = set()
for i in range(n - 2, -1, -1):
    curr = messages[i]
    email_subject = curr.get("email subject").strip()
    # Normalize the subject by removing the prefix
    if email_subject.startswith("Re: [cbioportal] "):
        email_subject = email_subject[17:].strip()
        if email_subject == new_messages[-1]["email subject"]:
            # Ensure that email text is a list
            if isinstance(new_messages[-1]["email text"], list):
                new_messages[-1]["email text"].append(curr["email text"])
            else:
                new_messages[-1]["email text"] = [new_messages[-1]["email text"], curr["email text"]]
    else:
        new_messages.append(curr)


write_json_file('mbox/combined_messages.json', new_messages) 
# print(new_messages)           
print(len(new_messages))
