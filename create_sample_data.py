import pandas as pd
import numpy as np

# Create sample fake news dataset
fake_data = {
    'title': [
        'Breaking: Secret Government Plot Revealed',
        'Shocking Celebrity Scandal That Will Blow Your Mind',
        'Miracle Cure Doctors Don\'t Want You to Know About',
        'Famous Person Dead - Conspiracy Theorists React',
        'NASA Covers Up Alien Contact Evidence'
    ] * 20,
    'text': [
        'According to anonymous sources, the government has been hiding the truth for decades. The evidence is overwhelming and undeniable.',
        'Inside sources reveal shocking details about celebrity. This will change everything you thought you knew.',
        'Medical experts hate this one simple trick. Patients are recovering miraculously after this discovery.',
        'The official story doesn\'t add up. Many believe there\'s more to the truth than what mainstream media reports.',
        'Whistleblowers come forward with classified documents proving extraterrestrial contact has been hidden.'
    ] * 20,
}

true_data = {
    'title': [
        'Market Analysis Shows Economic Growth',
        'Scientific Study Reveals New Treatment Breakthrough',
        'City Council Approves Infrastructure Development',
        'Researchers Publish Peer-Reviewed Findings',
        'Government Agency Releases Official Report'
    ] * 20,
    'text': [
        'Economists analyze quarterly data showing consistent growth across multiple sectors. The trend continues from previous quarters.',
        'After five years of research, scientists have published their findings in a reputable journal after rigorous peer review.',
        'The city council voted unanimously to approve the new infrastructure project that will benefit residents.',
        'The study involved 500 participants and was conducted according to standard research protocols.',
        'The official government report details the findings from their comprehensive investigation and analysis.'
    ] * 20,
}

fake_df = pd.DataFrame(fake_data)
true_df = pd.DataFrame(true_data)

fake_df.to_csv('fake.csv', index=False)
true_df.to_csv('true.csv', index=False)

print("Sample datasets created: fake.csv and true.csv")
