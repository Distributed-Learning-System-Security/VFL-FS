# Enter a parameter as the directory to be counted
cd $1
echo -e "The number of files: \c"
ls -l | grep "^-" | wc -l
echo -e "The number of directories: \c"
ls -l | grep "^d" | wc -l
# count=0
# for x in `ls ${dir}/*` 
# do     
#   filename=`basename ${x}`
#   if [ -d ${filename} ]
#   then
#     ${a}=${${a}+1}
#   fi
# done
# echo ${a}
