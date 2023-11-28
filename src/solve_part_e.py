from mixed_pdf_tools import pdf_norm_expon_mixed, rej_sample

pdf = lambda x: pdf_norm_expon_mixed(x, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6)
print(rej_sample(pdf, 10, 5.0, 5.6, 100)[0])