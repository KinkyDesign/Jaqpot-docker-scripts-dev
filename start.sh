for D in `find . -mindepth 1 -maxdepth 1 -type d`
do
   cd $D
   ./start.sh &
   cd ..
done
