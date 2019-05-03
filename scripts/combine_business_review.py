import pandas as pd
import json

# load business data
with open("../data/yelp_academic_dataset_business.json", 'r') as business:
    business_jsons = [json.loads(line) for line in business.readlines()]

# convert jsons to data
# for business: stars, business_id, latitude, longitude, name, states, categories, review_count
business = pd.DataFrame(business_jsons)[['business_id', 'stars', 'name', 'categories', 'latitude', 'longitude', 'state', 'review_count']]

# load review data
reviews = pd.read_table('../data/reviews_2017.tsv', sep='\t',
                        names=['business_id', 'cool', 'date', 'funny', 'likes', 'review_id', 'average_stars', 'text', 'useful', 'user_id'])

# join two datasets
combined_ = business.set_index('business_id').join(reviews.set_index('business_id'), how='inner')
combined_['business_id'] = combined_.index

# select 40% of businesses each class
combined_pos = combined_[combined_['average_stars'] >= 4]
combined_neg = combined_[combined_['average_stars'] < 4]
combined_pos = combined_pos[:int(combined_pos.shape[0]*0.4)]
combined_neg = combined_neg[:int(combined_neg.shape[0]*0.4)]
combined_ = pd.concat([combined_pos, combined_neg]).sample(frac=1)

# write to tsv
combined_.to_csv('../data/business_review_2017.tsv', sep='\t', index=False)