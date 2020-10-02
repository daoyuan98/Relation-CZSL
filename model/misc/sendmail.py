#!/usr/bin/env python3
import os
import argparse
import mimetypes
import datetime

try:
    from flask import Flask
    from flask_mail import Mail, Message
except ImportError:
    print('Please run \"pip install flask_mail\" first.')
    exit(-1)

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user-name', type=str, default=None)
parser.add_argument('-p', '--password', type=str, default=None)
parser.add_argument('-s', '--subject', type=str, required=True)
parser.add_argument('-r', '--recipient', type=str, required=True)
parser.add_argument('-b', '--body', type=str, default=' ')
parser.add_argument('-a', '--attachment', type=str, default=None)
parser.add_argument('-t', '--attachment_type', '--attachment-type', type=str, default=None)

parser.add_argument('--use-tls', action='store_true', default=False)
parser.add_argument('--no-ssl', action='store_true', default=False)
parser.add_argument('--server-port', type=int, default=465)
parser.add_argument('--server', type=str, default='smtp.gmail.com')

args = parser.parse_args()

if args.user_name is None:
    args.user_name = os.getenv('EMAILUSER')

if args.password is None:
    args.password = os.getenv('EMAILPASS')

app = Flask(__name__)
mail_settings = {
    "MAIL_SERVER": args.server,
    "MAIL_PORT": args.server_port,
    "MAIL_USE_TLS": args.use_tls,
    "MAIL_USE_SSL": not args.no_ssl,
    "MAIL_USERNAME": args.user_name,
    "MAIL_PASSWORD": args.password
}

app.config.update(mail_settings)
mail = Mail(app)

if __name__ == '__main__':
    with app.app_context():
        msg = Message(subject=args.subject,
                      sender=app.config.get("MAIL_USERNAME"),
                      recipients=[args.recipient],
                      body=args.body)
        if args.attachment:
            mime = mimetypes.MimeTypes()
            if not args.attachment_type:
                ftype = mime.guess_type(args.attachment)[0]
            else:
                ftype = args.attachment_type
            with app.open_resource(args.attachment) as fp:
                msg.attach(args.attachment, ftype, fp.read())
        mail.send(msg)
    print(str(datetime.datetime.now()) + ': Email sent successfully.')