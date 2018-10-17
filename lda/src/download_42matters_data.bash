#!/bin/bash -v

# old way
# aws s3 cp s3://ogury-apps-embedding-prod-eu-west-1/doc2vec/matters42/matters.en.2018-09.json.gz ../data/

# aws s3 ls s3://ogury-42matters-eu-west-1/1/42apps/v0.1/production/playstore/lookup-weekly/2018-08-31/ --human-readable
aws s3 cp s3://ogury-42matters-eu-west-1/1/42apps/v0.1/production/playstore/lookup-weekly/2018-08-31/playstore-00.tar.gz ../data/

exit 0

# count nb of apps
gzcat matters.en.2018-09.json.gz | grep -o short_desc | wc -l
gzcat matters.en.2018-09.json.gz | grep -o cat_key    | wc -l
