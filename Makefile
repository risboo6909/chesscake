Makefilebuild: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run: build
	.venv/bin/python3 src/main.py

daemonize: build
	nohup .venv/bin/python3 -u src/main.py >log 2>&1 &

kill:
	pkill -f chesscake
