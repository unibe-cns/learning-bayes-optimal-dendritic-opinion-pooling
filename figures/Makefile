all: fig4 fig6 fig7 fig8

fig4: fig4.pdf

fig4.pdf: fig4.py ../data/fig4/res.pkl
	python fig4.py

fig6: fig6.pdf

fig6.pdf: fig6.py figure_config.py ../data/fig6/res.pkl ../data/fig6/params.pkl
	python fig6.py

fig7: fig7.pdf

fig7.pdf: fig7.py figure_config.py ../data/fig7/res_test.pkl ../data/fig7/model_final.torch ../experiments/fig7_params.json
	python fig7.py

fig8: fig8.pdf

fig8.pdf: fig8.py ../data/fig8/res.pkl
	python fig8.py

.PHONY: fig4 fig6 fig7 fig8
