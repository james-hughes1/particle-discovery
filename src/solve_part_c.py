from mixed_pdf_tools import pdf_norm_expon_mixed, check_normalisation


def pdf_1(x):
    return pdf_norm_expon_mixed(
        x, f=0.05, la=0.5, mu=5.4, sg=0.1, alpha=5.0, beta=5.6
    )


def pdf_2(x):
    return pdf_norm_expon_mixed(
        x, f=0.9, la=0.5, mu=5.4, sg=0.01, alpha=5.0, beta=5.6
    )


def pdf_3(x):
    return pdf_norm_expon_mixed(
        x, f=0.1, la=2.0, mu=1.0, sg=1.0, alpha=5.0, beta=5.6
    )


print(
    "Integral when 1st set of parameters used:    {}".format(
        check_normalisation(pdf_1, 5.0, 5.6)
    )
)
print(
    "Integral when 2nd set of parameters used:    {}".format(
        check_normalisation(pdf_2, 5.0, 5.6)
    )
)
print(
    "Integral when 3rd set of parameters used:    {}".format(
        check_normalisation(pdf_3, 5.0, 5.6)
    )
)
