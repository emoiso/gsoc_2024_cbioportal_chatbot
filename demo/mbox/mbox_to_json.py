import mailbox
import bs4
import json


def get_html_text(html):
    try:
        return bs4.BeautifulSoup(html, 'lxml').body.get_text(' ', strip=True)
    except AttributeError:  # message contents empty
        return None


class GmailMboxMessage():
    def __init__(self, email_data):
        if not isinstance(email_data, mailbox.mboxMessage):
            raise TypeError('Variable must be type mailbox.mboxMessage')
        self.email_data = email_data

    def parse_email(self):
        email_labels = self.email_data['X-Gmail-Labels']
        email_date = self.email_data['Date']
        email_from = self.email_data['From']
        email_to = self.email_data['To']
        email_subject = self.email_data['Subject']
        email_text = self.read_email_payload() 

        self.email_from = self.email_data['From']
        email_to = self.email_data['To']
        self.email_to = self.email_data['To']
        email_subject = self.email_data['Subject']
        self.email_subject = self.email_data['Subject']
        email_text = self.read_email_payload() 
        self.email_text = self.read_email_payload()

    def read_email_payload(self):
        email_payload = self.email_data.get_payload()
        if self.email_data.is_multipart():
            email_messages = list(self._get_email_messages(email_payload))
        else:
            email_messages = [email_payload]
        return [self._read_email_text(msg) for msg in email_messages]

    def _get_email_messages(self, email_payload):
        for msg in email_payload:
            if isinstance(msg, (list, tuple)):
                for submsg in self._get_email_messages(msg):
                    yield submsg
            elif msg.is_multipart():
                for submsg in self._get_email_messages(msg.get_payload()):
                    yield submsg
            else:
                yield msg

    def _read_email_text(self, msg):
        content_type = 'NA' if isinstance(msg, str) else msg.get_content_type()
        encoding = 'NA' if isinstance(msg, str) else msg.get('Content-Transfer-Encoding', 'NA')
        if 'text/plain' in content_type and 'base64' not in encoding:
            msg_text = msg.get_payload()
        elif 'text/html' in content_type and 'base64' not in encoding:
            msg_text = get_html_text(msg.get_payload())
        elif content_type == 'NA':
            msg_text = get_html_text(msg)
        else:
            msg_text = None
        return (content_type, encoding, msg_text)


######################### End of library, example of use below

mbox_obj = mailbox.mbox('langchain_gsoc_main/cBioportal_data/documentation_site/mbox/google_conversations.mbox')

num_entries = len(mbox_obj)

with open("mbox_meassage.json", "w") as file:
    output = []
    for idx, email_obj in enumerate(mbox_obj):
        email_data = GmailMboxMessage(email_obj)
        email_data.parse_email()
        dict = {}
        dict['email from'] = str(email_data.email_from)
        dict['email to'] = str(email_data.email_to)
        dict['email subject'] = str(email_data.email_subject)
        dict['email text'] = str(email_data.email_text)
        output.append(dict)
    json_object = json.dumps(output, indent=4)
    file.write(json_object)
