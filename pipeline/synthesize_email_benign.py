#!/usr/bin/env python3
"""Stream E — benign email negatives (channel-confound guard for LLMail).

LLMail adds ~4000 email-channel INJECTION rows, all with email surface structure
(Subject line, greeting, body, sign-off). If EVERY email-structured row in the
corpus is injection, the tiny model learns "email structure ⇒ attack" and would
flag legitimate mail the user pastes to summarize. This generator supplies the
missing side: real-shaped, BENIGN emails (order/shipping/invoice/meeting/
newsletter/account) — exactly the mail people actually feed to an AI assistant.

Combinatorial templates × CorpusWriter dedup → diverse benign email text.
`channel="email"`, `is_synthetic=True` (train-only per honest-eval), EN + DE.
Bounded by CAP so it balances, not floods.

License: our own authored text. Stdlib-only. No network.
"""
from __future__ import annotations

import random

from corpus_common import CorpusWriter, Row

LICENSE = "own-synthetic-CC0"
CAP = 420
SEED = 42

# (subject, body-core) pairs per category; slots filled combinatorially.
EN = {
    "greet": ["Hi {name},", "Hello {name},", "Dear {name},", "Hi there,", "Good morning {name},"],
    "sign": ["Best regards,\n{sender}", "Kind regards,\n{sender}", "Thanks,\n{sender}",
             "Best,\nThe {org} Team", "Sincerely,\n{sender}"],
    "cat": [
        ("Your order {oid} has shipped",
         "Your order {oid} is on its way and should arrive by {date}. "
         "You can track it with the number {trk}. Thank you for shopping with {org}."),
        ("Invoice {inv} for {org}",
         "Please find attached invoice {inv} dated {date} for a total of €{amt}. "
         "Payment is due within 14 days to the account on file. Let us know if you have questions."),
        ("Meeting invitation: {topic}",
         "You're invited to a meeting about {topic} on {date} at {time}. "
         "The agenda covers the quarterly review and next steps. Please confirm your attendance."),
        ("{org} newsletter — {month}",
         "Here are this month's highlights: new features, a customer story, and upcoming events. "
         "You can update your preferences or unsubscribe at any time from your account settings."),
        ("Your {org} account statement",
         "Your monthly statement for {month} is now available. Your current balance is €{amt}. "
         "No action is needed; this message is for your records."),
        ("Appointment reminder",
         "This is a friendly reminder of your appointment on {date} at {time}. "
         "If you need to reschedule, reply to this email or call us during business hours."),
        ("Delivery scheduled for {date}",
         "Your package (order {oid}) is scheduled for delivery on {date}. "
         "Someone should be available to receive it. Tracking: {trk}."),
        ("Welcome to {org}",
         "Thanks for signing up. Your account is ready. Here are a few tips to get started, "
         "and our support team is happy to help if you have any questions."),
    ],
}
DE = {
    "greet": ["Hallo {name},", "Guten Tag {name},", "Sehr geehrte/r {name},", "Liebe/r {name},"],
    "sign": ["Mit freundlichen Grüßen,\n{sender}", "Beste Grüße,\n{sender}",
             "Vielen Dank,\n{sender}", "Ihr {org}-Team"],
    "cat": [
        ("Ihre Bestellung {oid} wurde versandt",
         "Ihre Bestellung {oid} ist unterwegs und sollte bis zum {date} ankommen. "
         "Sie können sie mit der Nummer {trk} verfolgen. Vielen Dank für Ihren Einkauf bei {org}."),
        ("Rechnung {inv} von {org}",
         "Anbei finden Sie die Rechnung {inv} vom {date} über einen Gesamtbetrag von €{amt}. "
         "Die Zahlung ist innerhalb von 14 Tagen fällig. Bei Fragen melden Sie sich gerne."),
        ("Einladung zum Termin: {topic}",
         "Wir laden Sie zu einem Termin zum Thema {topic} am {date} um {time} Uhr ein. "
         "Bitte bestätigen Sie Ihre Teilnahme."),
        ("Ihr {org}-Kontoauszug",
         "Ihr monatlicher Auszug für {month} ist jetzt verfügbar. Ihr aktueller Saldo beträgt €{amt}. "
         "Es ist keine Aktion erforderlich; diese Nachricht dient nur zu Ihrer Information."),
        ("Terminerinnerung",
         "Dies ist eine freundliche Erinnerung an Ihren Termin am {date} um {time} Uhr. "
         "Wenn Sie verschieben müssen, antworten Sie bitte auf diese E-Mail."),
    ],
}

NAMES = ["Anna", "Michael", "Sophie", "Thomas", "Laura", "David", "Ersin", "Maria", "Jonas", "customer"]
SENDERS = ["Customer Support", "The Billing Team", "Sarah from Support", "Account Services", "Your bank"]
ORGS = ["Zalando", "DHL", "Sparkasse", "Amazon", "Otto", "A1", "Volksbank", "MediaMarkt", "IKEA"]
TOPICS = ["the Q3 roadmap", "budget planning", "the new onboarding flow", "the marketing review"]
MONTHS = ["January", "March", "June", "October", "November"]
DATES = ["March 14", "June 3", "October 21", "November 9", "15.03.", "03.06.", "21.10."]


def gen(writer: CorpusWriter, block: dict, lang: str, rng: random.Random, want: int) -> int:
    made = 0
    tries = 0
    while made < want and tries < want * 40:
        tries += 1
        subj_t, body_t = rng.choice(block["cat"])
        slots = dict(
            name=rng.choice(NAMES), sender=rng.choice(SENDERS), org=rng.choice(ORGS),
            topic=rng.choice(TOPICS), month=rng.choice(MONTHS), date=rng.choice(DATES),
            time=f"{rng.randint(8,17)}:{rng.choice(['00','15','30','45'])}",
            oid=f"{rng.randint(10**6, 10**7)}", inv=f"INV-{rng.randint(1000,9999)}",
            trk=f"{rng.choice(['DE','AT'])}{rng.randint(10**8, 10**9)}",
            amt=f"{rng.randint(9, 899)}.{rng.randint(0,99):02d}",
        )
        subject = subj_t.format(**slots)
        body = body_t.format(**slots)
        greet = rng.choice(block["greet"]).format(**slots)
        sign = rng.choice(block["sign"]).format(**slots)
        text = f"Subject: {subject}\n\n{greet}\n\n{body}\n\n{sign}"
        if writer.add(Row(text=text, label="benign", source="email-benign",
                          license=LICENSE, lang=lang, channel="email",
                          is_synthetic=True, notes="channel-confound-guard")):
            made += 1
    return made


def main() -> int:
    rng = random.Random(SEED)
    writer = CorpusWriter()
    n_en = gen(writer, EN, "en", rng, int(CAP * 0.65))
    n_de = gen(writer, DE, "de", rng, CAP - n_en)
    writer.report("email-benign")
    print(f"[email-benign] en={n_en} de={n_de} (channel=email benign, train-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
