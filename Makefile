boot:
	python3 -m main

first_boot:
	python3 -m main
	mkdir -p runs
	gnome-terminal -- sh -c "tensorboard --logdir=runs"
	xdg-open http://localhost:6006
	
set-up_tensorboard:
	mkdir -p runs
	tensorboard --logdir=runs
	xdg-open http://localhost:6006

install:
	pip install -r requirements.txt