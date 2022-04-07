# Test with sphinx
pip install sphinx==1.8.6
pip install sphinx_rtd_theme

rm -rf build
make html

# upload to oss
ossconfig=`cat /home/admin/workspace/odps_clt_release_64/conf/atp-public-eki`
echo 'copy files to atp-modelzoo docs'
ossutil64 cp -f build oss://atp-modelzoo-sh/release/easynlp/easynlp_docs/ $ossconfig --recursive
