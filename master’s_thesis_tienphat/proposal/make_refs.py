#!/usr/bin/env python3
"""Write references.tex (IEEE style) in the order of first citation in proposal.tex.

Entries carry a DOI whenever one exists. Web pages carry the date they were checked.
Fails if a key is cited but not defined here.

    cd proposal && python3 make_refs.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCESSED = "Nov.~20, 2026"


def doi(d):
    return rf"doi: \href{{https://doi.org/{d}}}{{{d.replace('_', chr(92) + '_')}}}"


def web(url):
    return rf"Accessed: {ACCESSED}. [Online]. Available: \url{{{url}}}"


REFS = {
    "laroca2021efficient": "R. Laroca, L. A. Zanlorensi, G. R. Gon\\c{c}alves, E. Todt, W. R. Schwartz, and D. Menotti, ``An efficient and layout-independent automatic license plate recognition system based on the YOLO detector,'' \\emph{IET Intell. Transp. Syst.}, vol.~15, no.~4, pp.~483--503, 2021, " + doi("10.1049/itr2.12030") + ".",
    "shi2017crnn": "B. Shi, X. Bai, and C. Yao, ``An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition,'' \\emph{IEEE Trans. Pattern Anal. Mach. Intell.}, vol.~39, no.~11, pp.~2298--2304, 2017, " + doi("10.1109/TPAMI.2016.2646371") + ".",
    "ke2023ultrafast": "X. Ke, G. Zeng, and W. Guo, ``An ultra-fast automatic license plate recognition approach for unconstrained scenarios,'' \\emph{IEEE Trans. Intell. Transp. Syst.}, vol.~24, no.~5, pp.~5172--5185, 2023, " + doi("10.1109/TITS.2023.3237581") + ".",
    "laroca2026icpr": "R. Laroca \\emph{et al.}, ``ICPR 2026 competition on low-resolution license plate recognition,'' in \\emph{Proc. Int. Conf. Pattern Recognit. (ICPR)}, ser. Lecture Notes in Computer Science, vol.~16827. Springer, 2026, pp.~256--275, " + doi("10.1007/978-3-032-31936-4_18") + ".",
    "baselinecode": "MultiFrame-LPR: Baseline code of the ICPR 2026 LRLPR competition (ResTran). " + web("https://github.com/duongtruongbinh/MultiFrame-LPR"),
    "recht2019imagenet": "B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, ``Do ImageNet classifiers generalize to ImageNet?'' in \\emph{Proc. Int. Conf. Mach. Learn. (ICML)}, 2019, arXiv:1902.10811, " + doi("10.48550/arXiv.1902.10811") + ".",
    "smith2007tesseract": "R. Smith, ``An overview of the Tesseract OCR engine,'' in \\emph{Proc. 9th Int. Conf. Document Anal. Recognit. (ICDAR)}, Curitiba, Brazil, 2007, pp.~629--633, " + doi("10.1109/ICDAR.2007.4376991") + ".",
    "du2020ppocr": "Y. Du \\emph{et al.}, ``PP-OCR: A practical ultra lightweight OCR system,'' arXiv:2009.09941, 2020, " + doi("10.48550/arXiv.2009.09941") + ".",
    "nascimento2023sr": "V. Nascimento, R. Laroca, J. A. Lambert, W. R. Schwartz, and D. Menotti, ``Super-resolution of license plate images using attention modules and sub-pixel convolution layers,'' \\emph{Comput. Graph.}, vol.~113, pp.~69--76, 2023, " + doi("10.1016/j.cag.2023.05.005") + ".",
    "nascimento2025benchmark": "V. Nascimento, G. E. Lima, R. O. Ribeiro, W. R. Schwartz, R. Laroca, and D. Menotti, ``Toward advancing license plate super-resolution in real-world scenarios: A dataset and benchmark,'' \\emph{J. Braz. Comput. Soc.}, vol.~31, no.~1, pp.~435--448, 2025, " + doi("10.5753/jbcs.2025.5159") + ".",
    "wang2020tsrn": "W. Wang \\emph{et al.}, ``Scene text image super-resolution in the wild,'' in \\emph{Computer Vision -- ECCV 2020}, ser. Lecture Notes in Computer Science, vol.~12355. Springer, 2020, pp.~650--666, " + doi("10.1007/978-3-030-58607-2_38") + ".",
    "na2025mflpr2": "K. Na \\emph{et al.}, ``MF-LPR$^2$: Multi-frame license plate image restoration and recognition using optical flow,'' \\emph{Comput. Vis. Image Underst.}, vol.~256, Art. no.~104361, 2025, " + doi("10.1016/j.cviu.2025.104361") + ".",
    "jaderberg2015stn": "M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, ``Spatial transformer networks,'' in \\emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2015, arXiv:1506.02025, " + doi("10.48550/arXiv.1506.02025") + ".",
    "he2016resnet": "K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \\emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2016, pp.~770--778, " + doi("10.1109/CVPR.2016.90") + ".",
    "vaswani2017attention": "A. Vaswani \\emph{et al.}, ``Attention is all you need,'' in \\emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2017, arXiv:1706.03762, " + doi("10.48550/arXiv.1706.03762") + ".",
    "graves2006ctc": "A. Graves, S. Fern\\'andez, F. Gomez, and J. Schmidhuber, ``Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,'' in \\emph{Proc. 23rd Int. Conf. Mach. Learn. (ICML)}, 2006, pp.~369--376, " + doi("10.1145/1143844.1143891") + ".",
    "guo2017calibration": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, ``On calibration of modern neural networks,'' in \\emph{Proc. 34th Int. Conf. Mach. Learn. (ICML)}, PMLR vol.~70, 2017, pp.~1321--1330, " + doi("10.48550/arXiv.1706.04599") + ".",
    "geifman2017selective": "Y. Geifman and R. El-Yaniv, ``Selective classification for deep neural networks,'' in \\emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2017, arXiv:1705.08500, " + doi("10.48550/arXiv.1705.08500") + ".",
    "nosek2018preregistration": "B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, ``The preregistration revolution,'' \\emph{Proc. Natl. Acad. Sci. USA}, vol.~115, no.~11, pp.~2600--2606, 2018, " + doi("10.1073/pnas.1708274114") + ".",
    "fastapi": "S. Ram\\'irez, ``FastAPI documentation.'' " + web("https://fastapi.tiangolo.com"),
    "paszke2019pytorch": "A. Paszke \\emph{et al.}, ``PyTorch: An imperative style, high-performance deep learning library,'' in \\emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol.~32, 2019, arXiv:1912.01703, " + doi("10.48550/arXiv.1912.01703") + ".",
    "openapi": "OpenAPI Initiative, ``OpenAPI Specification, version 3.1.0,'' 2021. " + web("https://spec.openapis.org/oas/v3.1.0"),
    "abid2019gradio": "A. Abid, A. Abdalla, A. Abid, D. Khan, A. Alfozan, and J. Zou, ``Gradio: Hassle-free sharing and testing of ML models in the wild,'' presented at the \\emph{ICML Workshop on Human in the Loop Learning}, 2019, arXiv:1906.02569, " + doi("10.48550/arXiv.1906.02569") + ".",
    "dockercompose": "Docker Inc., ``Docker Compose documentation.'' " + web("https://docs.docker.com/compose/"),
    "cloudflaretunnel": "Cloudflare, Inc., ``Cloudflare Tunnel documentation.'' " + web("https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/"),
    "buslaev2020albumentations": "A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, ``Albumentations: Fast and flexible image augmentations,'' \\emph{Information}, vol.~11, no.~2, Art. no.~125, 2020, " + doi("10.3390/info11020125") + ".",
    "lee2015dsn": "C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, ``Deeply-supervised nets,'' in \\emph{Proc. Int. Conf. Artif. Intell. Statist. (AISTATS)}, 2015, arXiv:1409.5185, " + doi("10.48550/arXiv.1409.5185") + ".",
    "neverova2016moddrop": "N. Neverova, C. Wolf, G. Taylor, and F. Nebout, ``ModDrop: Adaptive multi-modal gesture recognition,'' \\emph{IEEE Trans. Pattern Anal. Mach. Intell.}, vol.~38, no.~8, pp.~1692--1706, 2016, " + doi("10.1109/TPAMI.2015.2461544") + ".",
    "efron1993bootstrap": "B. Efron and R. J. Tibshirani, \\emph{An Introduction to the Bootstrap}. New York, NY, USA: Chapman and Hall/CRC, 1993, " + doi("10.1201/9780429246593") + ".",
    "mcnemar1947": "Q. McNemar, ``Note on the sampling error of the difference between correlated proportions or percentages,'' \\emph{Psychometrika}, vol.~12, no.~2, pp.~153--157, 1947, " + doi("10.1007/BF02295996") + ".",
    "brooke1996sus": "J. Brooke, ``SUS: A `quick and dirty' usability scale,'' in \\emph{Usability Evaluation in Industry}, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. London, U.K.: Taylor \\& Francis, 1996, pp.~189--194.",
}


def main():
    tex = (HERE / "proposal.tex").read_text()
    body = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%"))
    order = []
    for group in re.findall(r"\\cite\{([^}]*)\}", body):
        for key in (k.strip() for k in group.split(",")):
            if key not in order:
                order.append(key)
    missing = [k for k in order if k not in REFS]
    if missing:
        sys.exit(f"missing references: {missing}")
    unused = [k for k in REFS if k not in order]
    lines = ["% Generated by make_refs.py; do not edit by hand.", "\\begin{thebibliography}{99}",
             "\\small\\setlength{\\itemsep}{2pt}"]
    lines += [f"\\bibitem{{{k}}} {REFS[k]}" for k in order]
    lines.append("\\end{thebibliography}")
    (HERE / "references.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(order)} references written; unused: {unused}")


if __name__ == "__main__":
    main()
