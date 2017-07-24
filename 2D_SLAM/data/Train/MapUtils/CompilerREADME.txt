Ubuntu Python 2:

1. Inside MapUtils/pybind_cpp/CMakeLists.txt, on line 8, change the python version:
	FIND_PACKAGE(PythonLibs 2 REQUIRED)

2. Inside MapUtils/pybind_cpp run:
	mkdir build && cd build && cmake .. && make && cd ..

3. Make sure the file you just created (MapUtils.so) is in the same folder as MapUtils.py

4. Run the test file:
	python testMapUtils.py

5. Output should be "Success!"



Ubuntu Python 3:

1. Inside MapUtils/pybind_cpp/CMakeLists.txt, on line 8, change the python version:
	FIND_PACKAGE(PythonLibs 3 REQUIRED)

2. Inside MapUtils/pybind_cpp run:
	mkdir build && cd build && cmake .. && make && cd ..

3. Make sure the file you just created (MapUtils.so) is in the same folder as MapUtils.py

4. Run the test file:
	python testMapUtils.py

5. Output should be "Success!"



Windows Python 2:

1. Download cygwin setup: https://cygwin.com/install.html

2. Follow step 1 from here: http://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/

3. Open Command prompt: WindowsKey+R, type cmd, Enter

4. Locate setup-86_64.exe

5. Run the following in command prompt: setup-x86_64.exe -P wget -P gcc-g++ -P make -P cmake -P diffutils -P libmpfr-devel -P libgmp-devel -P libmpc-devel -P python -P python-devel



6. cd /cygdrive/c/path/to/pybind_cpp

7. mkdir build && cd build && cmake .. && make && cd ..



8. Make sure the file you just created (MapUtils.so) is in the same folder as MapUtils.py

9. Run the test file:
	python testMapUtils.py
build
10. Output should be "Success!"


C:\Python27\include
C:\Python27\libs\libpython27.a

cd '/cygdrive/c/Users/Lenovo/Desktop/ESE 650/Project/PJ3/Proj3_2017_Train/MapUtils/pybind_cpp'
