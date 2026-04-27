#!/bin/bash

# 1. Password for the LOCAL pg_dump command
export PGPASSWORD="School#1607"

# 2. Supabase URL with URL-encoded password (%23 for # and %40 for @)
export SUPABASE_URL="postgresql://postgres.jsnczuqnlewxtdpcuurl:Iamanidiot%231607%23Unique%402026@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"

echo "Starting migration from local nep_db to Supabase..."

# 3. Dump the local database and pipe it directly to Supabase
pg_dump \
  -h localhost \
  -p 5432 \
  -U nep_admin \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  -d nep_db | psql "$SUPABASE_URL"

echo "Migration complete! Run python3 supabase_check.py again to verify."