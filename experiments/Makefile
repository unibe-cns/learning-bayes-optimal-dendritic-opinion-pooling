all: fig4 fig6 fig7 fig8

fig4: ../data/fig4/res.pkl

../data/fig4/res.pkl: fig4.py fig4_params.json ../data/fig7/model_final.torch
	python fig4.py

fig6: ../data/fig6/res.pkl ../data/fig6/params.pkl

../data/fig6/res.pkl ../data/fig6/params.pkl: fig6.py
	python fig6.py

fig7: ../data/fig7/res_test_VT.pkl ../data/fig7/res_test_V.pkl ../data/fig7/res_test_T.pkl

../data/fig7/res_test_VT.pkl ../data/fig7/res_test_V.pkl ../data/fig7/res_test_T.pkl: fig7_testing.py ../data/fig7/model_final.torch
	python fig7_testing.py

../data/fig7/model_final.torch: fig7_training.py fig7_params.json sim.py
	python fig7_training.py

fig8: ../data/fig8/res.pkl

../data/fig8/res.pkl: fig8.py ../data/fig7/model_final.torch
	python fig8.py

.PHONY: fig4 fig6 fig7 fig8
