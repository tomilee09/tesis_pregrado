all: main

main: main.py
	python3 main.py

run: 
	python3 main.py

tom: 
	python3 main.py 2

gabbo: 
	python3 main.py 1

clean: 
	rm -r __pycache__