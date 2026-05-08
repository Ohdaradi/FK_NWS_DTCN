import pandas as pd
import numpy as np

# Create LIAR dataset files
liar_labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
liar_cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'context', 'history1', 'history2', 'history3', 'history4', 'history5']

# Sample statements for each label
statements = {
    'pants-fire': [
        'The moon is made of cheese and visible from space',
        'Gravity was invented by scientists in 1985',
        'Water flows uphill naturally in most countries'
    ],
    'false': [
        'The Earth is completely flat',
        'Vaccines cause autism',
        'The sun revolves around the Earth'
    ],
    'barely-true': [
        'Most people like coffee',
        'Some animals can fly',
        'Most people sleep at night'
    ],
    'half-true': [
        'Climate is changing and humans contribute',
        'Some medications have side effects',
        'Technology has both benefits and drawbacks'
    ],
    'mostly-true': [
        'The Earth orbits the sun',
        'Vaccines prevent diseases',
        'Water is essential for life'
    ],
    'true': [
        'The Earth is round',
        'Oxygen is necessary for most living things',
        'The sun is a star'
    ]
}

# Create train, test, and valid TSV files
for dataset_type in ['train', 'test', 'valid']:
    rows = []
    for label, stmts in statements.items():
        for i, stmt in enumerate(stmts * 10):  # Repeat to get more data
            rows.append({
                'id': f'{dataset_type}_{label}_{i}',
                'label': label,
                'statement': stmt,
                'subject': 'general',
                'speaker': 'news',
                'job': 'journalist',
                'state': 'N/A',
                'party': 'N/A',
                'context': 'news article',
                'history1': '0',
                'history2': '0',
                'history3': '0',
                'history4': '0',
                'history5': '0'
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(f'{dataset_type}.tsv', sep='\t', index=False, header=False)
    print(f"Created {dataset_type}.tsv with {len(df)} rows")

print("LIAR dataset files created successfully!")
