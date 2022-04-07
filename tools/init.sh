#!/bin/bash

# init pre-commit check hook
if [ ! -d .git/hooks/ ]; then
    mkdir .git/hooks
fi

rm -rf .git/hooks/pre-commit
cp tools/pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

# other inits
#python git-lfs/git_lfs.py pull

# compile proto files
#source scripts/gen_proto.sh

