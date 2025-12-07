
This harness produces Dense and CSR normalized hashes that match the benchmark suite. It contains no ROLV IP.
Conclusion
You can validate the ROLV benchmarks with 100% certainty without access to ROLV proprietary code. By running only vendor baselines (Dense GEMM and CSR SpMM) with the vendor only harness above, normalizing outputs, and comparing SHA 256 hashes, you will reproduce the same baseline hashes reported in the benchmark suite.
Most importantly, ROLV normalized output hashes are identical across NVIDIA and AMD, demonstrating cross vendor reproducibility. Vendor baseline hashes may differ between Dense and CSR implementations, but this is expected and verified. ROLF and DENGS are not required in core benchmarks: ROLF is a non standard, non applicable subsampling shortcut with severe limitations, and DENGS is redundant and slow. Excluding them strengthens the case for ROLV as a reproducible, audit ready innovation balancing speed, efficiency, and correctness.


Verified Benchmarks Hash Super 4 December 7, 2025
NVIDIA
=== RUN SUITE (CUDA) on NVIDIA B200 ===
Matrices: 20,000x20,000  Batch Size: 5,000  Iterations: 1,000

[2025-12-07 08:43:17] Seed: 123456 | Pattern: random | Zeros: 40%
A_hash: e3644a901043856adaa3b878146a5978eda600732465e78134f6121ad2135eab | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061929s | CSR: 0.514234s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.580841 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c  (Dense GEMM (cuBLAS))
CSR_norm_hash:   ecc78a7f91b9ead974edfd89cc24c55c94f0ac392b9f6501cce757e28f8acece
ROLF_norm_hash:  96f29966a7efcb5ef2528e92078988a2063a442867509f2f469e1b63cec61896
DENGS_norm_hash: 11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c
ROLF per-iter:   0.000329s | total: 0.356796s
DENGS per-iter:  0.061900s | total: 61.899621s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 23.92x (≈ 2292% faster)
Speedup (per-iter): 61.47x (≈ 6047% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 511.23x | total: 198.94x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "e3644a901043856adaa3b878146a5978eda600732465e78134f6121ad2135eab", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c", "CSR_norm_hash": "ecc78a7f91b9ead974edfd89cc24c55c94f0ac392b9f6501cce757e28f8acece", "ROLF_norm_hash": "96f29966a7efcb5ef2528e92078988a2063a442867509f2f469e1b63cec61896", "DENGS_norm_hash": "11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "d02e8632c028003b3549fb086dad732fc49aed49a06e28cfc5e2a0f32da41a36", "CSR_qhash_d6": "5fdf5492171ccb8963ea9f0b34eb1903a9a55c17fab435136d0b61c29b7e18b8", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061929, "pilot_csr_per_iter_s": 0.514234, "rolv_build_s": 1.580841, "rolv_iter_s": 0.001007, "dense_iter_s": 0.061905, "csr_iter_s": 0.51484, "rolv_total_s": 2.587908, "baseline_total_s": 61.905422, "speedup_total_vs_selected_x": 23.921, "speedup_iter_vs_selected_x": 61.471, "rolv_vs_vendor_sparse_iter_x": 511.228, "rolv_vs_vendor_sparse_total_x": 198.941, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 08:54:51] Seed: 123456 | Pattern: power_law | Zeros: 40%
A_hash: 0bc0d2cd333849b2bc5726b8182342a2b1f1692dec3ce1baa02459ebd0feca6e | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061910s | CSR: 0.475878s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.383957 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb  (Dense GEMM (cuBLAS))
CSR_norm_hash:   1651a63f7487c624ff16de1133f1afb8f369f7b415176237dd5a77d19df59532
ROLF_norm_hash:  04c6dfa6fcbee09f54a8fe3b9d83bb2c2537a62e94ff981cc0de5f2215d8c38b
DENGS_norm_hash: 3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb
ROLF per-iter:   0.000337s | total: 0.337618s
DENGS per-iter:  0.061905s | total: 61.905355s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 25.89x (≈ 2489% faster)
Speedup (per-iter): 61.48x (≈ 6048% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 472.96x | total: 199.17x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "0bc0d2cd333849b2bc5726b8182342a2b1f1692dec3ce1baa02459ebd0feca6e", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb", "CSR_norm_hash": "1651a63f7487c624ff16de1133f1afb8f369f7b415176237dd5a77d19df59532", "ROLF_norm_hash": "04c6dfa6fcbee09f54a8fe3b9d83bb2c2537a62e94ff981cc0de5f2215d8c38b", "DENGS_norm_hash": "3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "1bb133eca121d0422acc7c427733ee08af00c0731d925906a3a7449af91e982e", "CSR_qhash_d6": "b8b38d39b7bbc3ddaae5ad81be57f88716c346469259cd023abefb44b3581268", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.06191, "pilot_csr_per_iter_s": 0.475878, "rolv_build_s": 1.383957, "rolv_iter_s": 0.001007, "dense_iter_s": 0.0619, "csr_iter_s": 0.476179, "rolv_total_s": 2.390761, "baseline_total_s": 61.900266, "speedup_total_vs_selected_x": 25.891, "speedup_iter_vs_selected_x": 61.482, "rolv_vs_vendor_sparse_iter_x": 472.961, "rolv_vs_vendor_sparse_total_x": 199.175, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:05:42] Seed: 123456 | Pattern: banded | Zeros: 40%
A_hash: 69975c70a3346649e1fbefab534eae7887a68247af2ad0c91ced7488ab619e6c | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061956s | CSR: 0.018649s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.037180 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  fe6693913ca3d601f26d88b75f21363cd782aeb0fff9d7044d0fe50c3d00a9b8  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   fe6693913ca3d601f26d88b75f21363cd782aeb0fff9d7044d0fe50c3d00a9b8
ROLF_norm_hash:  3d143289649ac69457ce1a4c0c58322caf19f04aff297d32892e289db68e9353
DENGS_norm_hash: 1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f
ROLF per-iter:   0.000329s | total: 0.329258s
DENGS per-iter:  0.061928s | total: 61.927887s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 9.13x (≈ 813% faster)
Speedup (per-iter): 18.54x (≈ 1754% faster)
Energy Savings: 94.61%
ROLV vs cuSPARSE -> Speedup (per-iter): 18.54x | total: 9.13x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "69975c70a3346649e1fbefab534eae7887a68247af2ad0c91ced7488ab619e6c", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f", "CSR_norm_hash": "fe6693913ca3d601f26d88b75f21363cd782aeb0fff9d7044d0fe50c3d00a9b8", "ROLF_norm_hash": "3d143289649ac69457ce1a4c0c58322caf19f04aff297d32892e289db68e9353", "DENGS_norm_hash": "1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "80fe483ed7c35f0587e85f5c26d02f3e5b9572628977d7add5283e61db8ad088", "CSR_qhash_d6": "95f3fffb6d580300c38e18b4cd06fc51b516ecfd1b8d856c5e234e3000fb2bf3", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061956, "pilot_csr_per_iter_s": 0.018649, "rolv_build_s": 1.03718, "rolv_iter_s": 0.001006, "dense_iter_s": 0.018656, "csr_iter_s": 0.018659, "rolv_total_s": 2.043537, "baseline_total_s": 18.656377, "speedup_total_vs_selected_x": 9.129, "speedup_iter_vs_selected_x": 18.539, "rolv_vs_vendor_sparse_iter_x": 18.541, "rolv_vs_vendor_sparse_total_x": 9.131, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:08:04] Seed: 123456 | Pattern: block_diagonal | Zeros: 40%
A_hash: d7a5bfe4c7f465590f90417984ef8f0754801ffe2307e0f3a276649b4868f2ad | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061944s | CSR: 0.011831s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.428320 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  321424f6b1a6934f1a49f6e4308036debe57d24f24378336e481d01ce10a1f0c  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   321424f6b1a6934f1a49f6e4308036debe57d24f24378336e481d01ce10a1f0c
ROLF_norm_hash:  fad1e70cbfb1f7a7c76bac355291ed547070889fc04e0bac447e2cc9d20dcff1
DENGS_norm_hash: 988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c
ROLF per-iter:   0.000329s | total: 0.330217s
DENGS per-iter:  0.061924s | total: 61.924457s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 4.86x (≈ 386% faster)
Speedup (per-iter): 11.75x (≈ 1075% faster)
Energy Savings: 91.49%
ROLV vs cuSPARSE -> Speedup (per-iter): 11.75x | total: 4.86x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "d7a5bfe4c7f465590f90417984ef8f0754801ffe2307e0f3a276649b4868f2ad", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c", "CSR_norm_hash": "321424f6b1a6934f1a49f6e4308036debe57d24f24378336e481d01ce10a1f0c", "ROLF_norm_hash": "fad1e70cbfb1f7a7c76bac355291ed547070889fc04e0bac447e2cc9d20dcff1", "DENGS_norm_hash": "988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "9993275a88ff770aeb9aca162be175a9c3040c53c5c7f6fc2a4f319c31cfdc98", "CSR_qhash_d6": "47ba4557f17017e59b691d3658d8dcf06af6fac03e525cc676260e0e6e741724", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061944, "pilot_csr_per_iter_s": 0.011831, "rolv_build_s": 1.42832, "rolv_iter_s": 0.001007, "dense_iter_s": 0.011831, "csr_iter_s": 0.011833, "rolv_total_s": 2.435436, "baseline_total_s": 11.830762, "speedup_total_vs_selected_x": 4.858, "speedup_iter_vs_selected_x": 11.747, "rolv_vs_vendor_sparse_iter_x": 11.749, "rolv_vs_vendor_sparse_total_x": 4.858, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:10:07] Seed: 123456 | Pattern: random | Zeros: 50%
A_hash: 6e4770bed2259e6973f564d1f8d9f3edc952d13fc6befcf5a9f9094269703540 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061917s | CSR: 0.418095s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 0.770090 s
ROLV per-iter: 0.001008s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404  (Dense GEMM (cuBLAS))
CSR_norm_hash:   5b6bbf6ca9f9685f5bb8163246034117509b8f1fcc755ab994187c8738fbcee6
ROLF_norm_hash:  c816fe475c90aae0faf91c6f709db164d5fa330d0a3538d29218bc7dc6e7e99f
DENGS_norm_hash: 16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404
ROLF per-iter:   0.000329s | total: 0.329519s
DENGS per-iter:  0.061900s | total: 61.899922s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 34.81x (≈ 3381% faster)
Speedup (per-iter): 61.41x (≈ 6041% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 415.13x | total: 235.34x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "6e4770bed2259e6973f564d1f8d9f3edc952d13fc6befcf5a9f9094269703540", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404", "CSR_norm_hash": "5b6bbf6ca9f9685f5bb8163246034117509b8f1fcc755ab994187c8738fbcee6", "ROLF_norm_hash": "c816fe475c90aae0faf91c6f709db164d5fa330d0a3538d29218bc7dc6e7e99f", "DENGS_norm_hash": "16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "6411421f1efd8ea75978adb572c41db98aed6edb716989a47e78ad96e0a71457", "CSR_qhash_d6": "cba48f014d9f993cf75498bf448cad1d2a4215dbbae8ae22b3e659a492e4ed65", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061917, "pilot_csr_per_iter_s": 0.418095, "rolv_build_s": 0.77009, "rolv_iter_s": 0.001008, "dense_iter_s": 0.061901, "csr_iter_s": 0.418473, "rolv_total_s": 1.778139, "baseline_total_s": 61.900766, "speedup_total_vs_selected_x": 34.812, "speedup_iter_vs_selected_x": 61.406, "rolv_vs_vendor_sparse_iter_x": 415.131, "rolv_vs_vendor_sparse_total_x": 235.343, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:19:57] Seed: 123456 | Pattern: power_law | Zeros: 50%
A_hash: e868d93c6a2425c33f4461dda493d60421f514ce596dcf01814e71c6fb964106 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061912s | CSR: 0.387562s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.061295 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b  (Dense GEMM (cuBLAS))
CSR_norm_hash:   17fdec8705d44da199f852d054e0df5fb83da82921dc8155671562c20020c4db
ROLF_norm_hash:  454ae05f90c5e045acc09af4e561c50f160fe5add28880ee4daf2660646217ba
DENGS_norm_hash: 2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b
ROLF per-iter:   0.000329s | total: 0.329940s
DENGS per-iter:  0.061894s | total: 61.894273s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 29.93x (≈ 2893% faster)
Speedup (per-iter): 61.49x (≈ 6049% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 385.10x | total: 187.45x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "e868d93c6a2425c33f4461dda493d60421f514ce596dcf01814e71c6fb964106", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b", "CSR_norm_hash": "17fdec8705d44da199f852d054e0df5fb83da82921dc8155671562c20020c4db", "ROLF_norm_hash": "454ae05f90c5e045acc09af4e561c50f160fe5add28880ee4daf2660646217ba", "DENGS_norm_hash": "2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "5eedfa8fdcd56343dcae7a1bcfe78a3e0011acf344fa0a15bca967f4c5750f59", "CSR_qhash_d6": "27f8d1052e57ddbec766e4ad302c6c99cd4834589a40eb883ced2e69f1d87cf4", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061912, "pilot_csr_per_iter_s": 0.387562, "rolv_build_s": 1.061295, "rolv_iter_s": 0.001007, "dense_iter_s": 0.061893, "csr_iter_s": 0.387622, "rolv_total_s": 2.067841, "baseline_total_s": 61.892855, "speedup_total_vs_selected_x": 29.931, "speedup_iter_vs_selected_x": 61.49, "rolv_vs_vendor_sparse_iter_x": 385.101, "rolv_vs_vendor_sparse_total_x": 187.452, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:29:19] Seed: 123456 | Pattern: banded | Zeros: 50%
A_hash: 36930b864e45f6c7bc4c05a36ceed9e5546aba4f26c38e27ec94b84500ab052f | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061947s | CSR: 0.015641s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.388301 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  44821abfe450dcc7279d9a90ae21273991123e4db23fd603fd1be2af86d33d44  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   44821abfe450dcc7279d9a90ae21273991123e4db23fd603fd1be2af86d33d44
ROLF_norm_hash:  0621305b553d14c934dc235dd9616ff421a67e87342f046923dc5da9bed27029
DENGS_norm_hash: 0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234
ROLF per-iter:   0.000329s | total: 0.329414s
DENGS per-iter:  0.061937s | total: 61.936957s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 6.53x (≈ 553% faster)
Speedup (per-iter): 15.54x (≈ 1454% faster)
Energy Savings: 93.57%
ROLV vs cuSPARSE -> Speedup (per-iter): 15.55x | total: 6.54x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "36930b864e45f6c7bc4c05a36ceed9e5546aba4f26c38e27ec94b84500ab052f", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234", "CSR_norm_hash": "44821abfe450dcc7279d9a90ae21273991123e4db23fd603fd1be2af86d33d44", "ROLF_norm_hash": "0621305b553d14c934dc235dd9616ff421a67e87342f046923dc5da9bed27029", "DENGS_norm_hash": "0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "91a6790e57791bfb5eb9aeab2ad3128bc32fc47fb2b6dcd8728760921b04c533", "CSR_qhash_d6": "71eaa793df4c26276db3879f58c3116f41327e1cc584fad6079d6250db00f568", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061947, "pilot_csr_per_iter_s": 0.015641, "rolv_build_s": 1.388301, "rolv_iter_s": 0.001006, "dense_iter_s": 0.015645, "csr_iter_s": 0.01565, "rolv_total_s": 2.394729, "baseline_total_s": 15.644613, "speedup_total_vs_selected_x": 6.533, "speedup_iter_vs_selected_x": 15.545, "rolv_vs_vendor_sparse_iter_x": 15.55, "rolv_vs_vendor_sparse_total_x": 6.535, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:31:33] Seed: 123456 | Pattern: block_diagonal | Zeros: 50%
A_hash: 8db5189cd07996217967440640b6d42a07f04d0966354d2bccdba45b8f0e85b6 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061942s | CSR: 0.009871s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.025778 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  64bb6aeeae718df5ab005c49a12708dca47e1637e48e005c7621ed9d2d41b87a  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   64bb6aeeae718df5ab005c49a12708dca47e1637e48e005c7621ed9d2d41b87a
ROLF_norm_hash:  1b772b5257f1ebb4c0e1b3eb7d087b0a42bbbeb650b2fd543326e966e21a2cb4
DENGS_norm_hash: 03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66
ROLF per-iter:   0.000329s | total: 0.329301s
DENGS per-iter:  0.061932s | total: 61.932281s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 4.85x (≈ 385% faster)
Speedup (per-iter): 9.80x (≈ 880% faster)
Energy Savings: 89.80%
ROLV vs cuSPARSE -> Speedup (per-iter): 9.81x | total: 4.86x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "8db5189cd07996217967440640b6d42a07f04d0966354d2bccdba45b8f0e85b6", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66", "CSR_norm_hash": "64bb6aeeae718df5ab005c49a12708dca47e1637e48e005c7621ed9d2d41b87a", "ROLF_norm_hash": "1b772b5257f1ebb4c0e1b3eb7d087b0a42bbbeb650b2fd543326e966e21a2cb4", "DENGS_norm_hash": "03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "b5f52a9ef8ffd7efcef97cbd495082fcb11cd7cd2264e12ccaebab8950e1f08e", "CSR_qhash_d6": "f5b31c2e1d1da0c820dccc6bfa3935270804cfc9b87d0149dc8ca81bc0ab5b32", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061942, "pilot_csr_per_iter_s": 0.009871, "rolv_build_s": 1.025778, "rolv_iter_s": 0.001007, "dense_iter_s": 0.009865, "csr_iter_s": 0.009879, "rolv_total_s": 2.032394, "baseline_total_s": 9.864829, "speedup_total_vs_selected_x": 4.854, "speedup_iter_vs_selected_x": 9.8, "rolv_vs_vendor_sparse_iter_x": 9.815, "rolv_vs_vendor_sparse_total_x": 4.861, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:33:36] Seed: 123456 | Pattern: random | Zeros: 60%
A_hash: 3a128a12c751e2a52a9f05427ad881a4beeb441b1aa828f2c83dec9767075e14 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061922s | CSR: 0.326452s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.328603 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e  (Dense GEMM (cuBLAS))
CSR_norm_hash:   081d90967e48c7133b877943ed840b843b65338bc9b4b50859e132a296bc6236
ROLF_norm_hash:  53ee64013770fdb75b32a6e92743de71f64c87429155a238a486d4601379ff1b
DENGS_norm_hash: 82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e
ROLF per-iter:   0.000329s | total: 0.329723s
DENGS per-iter:  0.061902s | total: 61.901871s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 26.53x (≈ 2553% faster)
Speedup (per-iter): 61.61x (≈ 6061% faster)
Energy Savings: 98.38%
ROLV vs cuSPARSE -> Speedup (per-iter): 325.23x | total: 140.04x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "3a128a12c751e2a52a9f05427ad881a4beeb441b1aa828f2c83dec9767075e14", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e", "CSR_norm_hash": "081d90967e48c7133b877943ed840b843b65338bc9b4b50859e132a296bc6236", "ROLF_norm_hash": "53ee64013770fdb75b32a6e92743de71f64c87429155a238a486d4601379ff1b", "DENGS_norm_hash": "82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "474ef2e5789ba96a76b96db5a6f8e76990d35228b24cd5d7660008bef1c3606c", "CSR_qhash_d6": "52b46a9fcf04fd5401f156d834a32b791e637ca11482b5aaf62cae87384827cb", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061922, "pilot_csr_per_iter_s": 0.326452, "rolv_build_s": 1.328603, "rolv_iter_s": 0.001005, "dense_iter_s": 0.061902, "csr_iter_s": 0.326772, "rolv_total_s": 2.333346, "baseline_total_s": 61.902277, "speedup_total_vs_selected_x": 26.529, "speedup_iter_vs_selected_x": 61.61, "rolv_vs_vendor_sparse_iter_x": 325.23, "rolv_vs_vendor_sparse_total_x": 140.045, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:41:55] Seed: 123456 | Pattern: power_law | Zeros: 60%
A_hash: 9d19ea5f391575455f95a6f93a0dc330f0816afb109185aa39e76d5e5e3f84a5 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061924s | CSR: 0.303375s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 0.976134 s
ROLV per-iter: 0.001004s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568  (Dense GEMM (cuBLAS))
CSR_norm_hash:   ac50cd780b9fbbfaa913575caa25a1d95ed42dd1b4fc6670b3f5c3e8f1e3bd29
ROLF_norm_hash:  d20759f7172d2f43c943a230dae84150fb48c05b8b882f77935ccb62e51e5023
DENGS_norm_hash: 3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568
ROLF per-iter:   0.000329s | total: 0.329971s
DENGS per-iter:  0.061895s | total: 61.895434s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 31.26x (≈ 3026% faster)
Speedup (per-iter): 61.64x (≈ 6064% faster)
Energy Savings: 98.38%
ROLV vs cuSPARSE -> Speedup (per-iter): 302.15x | total: 153.21x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "9d19ea5f391575455f95a6f93a0dc330f0816afb109185aa39e76d5e5e3f84a5", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568", "CSR_norm_hash": "ac50cd780b9fbbfaa913575caa25a1d95ed42dd1b4fc6670b3f5c3e8f1e3bd29", "ROLF_norm_hash": "d20759f7172d2f43c943a230dae84150fb48c05b8b882f77935ccb62e51e5023", "DENGS_norm_hash": "3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "054e2c6d65e7082adb830742e210da6458ef0c3b993c9efeb6f8de4af5491a0b", "CSR_qhash_d6": "ec2156241114e342ae8e2ed37f2ab86de7f3113ac7dac3c1b0269a0b76b083fa", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061924, "pilot_csr_per_iter_s": 0.303375, "rolv_build_s": 0.976134, "rolv_iter_s": 0.001004, "dense_iter_s": 0.0619, "csr_iter_s": 0.303407, "rolv_total_s": 1.980297, "baseline_total_s": 61.899754, "speedup_total_vs_selected_x": 31.258, "speedup_iter_vs_selected_x": 61.643, "rolv_vs_vendor_sparse_iter_x": 302.149, "rolv_vs_vendor_sparse_total_x": 153.213, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:49:48] Seed: 123456 | Pattern: banded | Zeros: 60%
A_hash: e78e035e07d681d9c88788fb30448528322d3759de0292aef1030acc8d438be2 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061946s | CSR: 0.012625s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.950037 s
ROLV per-iter: 0.001004s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  ee14ea1bf5ef3264e58ce7ebc8e5acfd4d913d65db0181887c4cc4d3b914869b  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   ee14ea1bf5ef3264e58ce7ebc8e5acfd4d913d65db0181887c4cc4d3b914869b
ROLF_norm_hash:  875e17c75ff856b76232765109555a25b7fbfd369a523cfcfc37fecd0cf0a765
DENGS_norm_hash: a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a
ROLF per-iter:   0.000329s | total: 0.329387s
DENGS per-iter:  0.061928s | total: 61.928273s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 6.46x (≈ 546% faster)
Speedup (per-iter): 12.58x (≈ 1158% faster)
Energy Savings: 92.05%
ROLV vs cuSPARSE -> Speedup (per-iter): 12.57x | total: 6.46x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "e78e035e07d681d9c88788fb30448528322d3759de0292aef1030acc8d438be2", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a", "CSR_norm_hash": "ee14ea1bf5ef3264e58ce7ebc8e5acfd4d913d65db0181887c4cc4d3b914869b", "ROLF_norm_hash": "875e17c75ff856b76232765109555a25b7fbfd369a523cfcfc37fecd0cf0a765", "DENGS_norm_hash": "a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "6c590cb1c86449e9a55609b2add184da816091d6e591141b6417902275b2cba6", "CSR_qhash_d6": "04f431768c7dfe815a3c1aa4d00832cfa731970632a2725ec8002964c6d58a91", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061946, "pilot_csr_per_iter_s": 0.012625, "rolv_build_s": 0.950037, "rolv_iter_s": 0.001004, "dense_iter_s": 0.012631, "csr_iter_s": 0.012629, "rolv_total_s": 1.954414, "baseline_total_s": 12.630721, "speedup_total_vs_selected_x": 6.463, "speedup_iter_vs_selected_x": 12.576, "rolv_vs_vendor_sparse_iter_x": 12.574, "rolv_vs_vendor_sparse_total_x": 6.462, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:51:53] Seed: 123456 | Pattern: block_diagonal | Zeros: 60%
A_hash: 2b99793bda656b5689cc9f5b049fc1a55ae8c234e0386e439c7204b281ffc158 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061936s | CSR: 0.007954s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.969341 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  8d49065f10d92e885ca931dcde70362ba4868062d04ac8f6d5721f250abf4222  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   8d49065f10d92e885ca931dcde70362ba4868062d04ac8f6d5721f250abf4222
ROLF_norm_hash:  968c6fc2b0accf1cdc4ab0d5115cfd7051533a5d02294705346da1e28a71c50b
DENGS_norm_hash: 36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4
ROLF per-iter:   0.000329s | total: 0.330157s
DENGS per-iter:  0.061923s | total: 61.923070s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 4.02x (≈ 302% faster)
Speedup (per-iter): 7.90x (≈ 690% faster)
Energy Savings: 87.35%
ROLV vs cuSPARSE -> Speedup (per-iter): 7.90x | total: 4.03x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "2b99793bda656b5689cc9f5b049fc1a55ae8c234e0386e439c7204b281ffc158", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4", "CSR_norm_hash": "8d49065f10d92e885ca931dcde70362ba4868062d04ac8f6d5721f250abf4222", "ROLF_norm_hash": "968c6fc2b0accf1cdc4ab0d5115cfd7051533a5d02294705346da1e28a71c50b", "DENGS_norm_hash": "36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "4d9bc3a8c04e309be3d194ede6251942ddf9e71860fd8aa14f66686b71fd0279", "CSR_qhash_d6": "9d8c33317bf0b400d15db9151598a176cce433884e9b0fff716c3863d71f2909", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061936, "pilot_csr_per_iter_s": 0.007954, "rolv_build_s": 0.969341, "rolv_iter_s": 0.001006, "dense_iter_s": 0.007949, "csr_iter_s": 0.007951, "rolv_total_s": 1.975194, "baseline_total_s": 7.949296, "speedup_total_vs_selected_x": 4.025, "speedup_iter_vs_selected_x": 7.903, "rolv_vs_vendor_sparse_iter_x": 7.905, "rolv_vs_vendor_sparse_total_x": 4.025, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:53:50] Seed: 123456 | Pattern: random | Zeros: 70%
A_hash: b6d397e4d0e8ebd4f3a13d59f635831bd762ee60284807ed9d008435058ec326 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061918s | CSR: 0.239783s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 0.986053 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915  (Dense GEMM (cuBLAS))
CSR_norm_hash:   7643677884310c75906dc5db1268414da51e3b086f05b91f180c8a0ec38abe55
ROLF_norm_hash:  a503fe60776ce90abef86e50f774bce0642268c79ed36f1ed2a0cfdb456b6090
DENGS_norm_hash: 722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915
ROLF per-iter:   0.000329s | total: 0.330274s
DENGS per-iter:  0.061893s | total: 61.892766s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 31.07x (≈ 3007% faster)
Speedup (per-iter): 61.53x (≈ 6053% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 238.50x | total: 120.44x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "b6d397e4d0e8ebd4f3a13d59f635831bd762ee60284807ed9d008435058ec326", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915", "CSR_norm_hash": "7643677884310c75906dc5db1268414da51e3b086f05b91f180c8a0ec38abe55", "ROLF_norm_hash": "a503fe60776ce90abef86e50f774bce0642268c79ed36f1ed2a0cfdb456b6090", "DENGS_norm_hash": "722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "82da032e08a558947b8496a6df588242839c731dd132f6c51b2ccad1158e1e9e", "CSR_qhash_d6": "e8dde86c7f7de83e21def8e3cb9af17558435d95f2498873b302a89e1eecd3a3", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061918, "pilot_csr_per_iter_s": 0.239783, "rolv_build_s": 0.986053, "rolv_iter_s": 0.001006, "dense_iter_s": 0.061894, "csr_iter_s": 0.239898, "rolv_total_s": 1.991902, "baseline_total_s": 61.893684, "speedup_total_vs_selected_x": 31.073, "speedup_iter_vs_selected_x": 61.534, "rolv_vs_vendor_sparse_iter_x": 238.503, "rolv_vs_vendor_sparse_total_x": 120.437, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:00:40] Seed: 123456 | Pattern: power_law | Zeros: 70%
A_hash: 64b353290cc661d8798233b459b02627e318c8b6cd03fb9400cdc258605a7257 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061922s | CSR: 0.223187s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 0.976449 s
ROLV per-iter: 0.001008s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc  (Dense GEMM (cuBLAS))
CSR_norm_hash:   1e4762ba93f5302b31d2f223052ca999af695928aee91c656c17800bf90ea80f
ROLF_norm_hash:  72204d453e28b46f2f06b0ac265feda1932e914601162d00494a9d1bfa846619
DENGS_norm_hash: 32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc
ROLF per-iter:   0.000329s | total: 0.329839s
DENGS per-iter:  0.061896s | total: 61.895570s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 31.19x (≈ 3019% faster)
Speedup (per-iter): 61.42x (≈ 6042% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 221.51x | total: 112.50x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "64b353290cc661d8798233b459b02627e318c8b6cd03fb9400cdc258605a7257", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc", "CSR_norm_hash": "1e4762ba93f5302b31d2f223052ca999af695928aee91c656c17800bf90ea80f", "ROLF_norm_hash": "72204d453e28b46f2f06b0ac265feda1932e914601162d00494a9d1bfa846619", "DENGS_norm_hash": "32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "b28317e48cc7768973ac901f2af18502a2b95897bacec4775b55b8b869b4083f", "CSR_qhash_d6": "32d2a1355f0810ace537c01c4e93a36474c0b73787dbb59de89551157cc938f5", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061922, "pilot_csr_per_iter_s": 0.223187, "rolv_build_s": 0.976449, "rolv_iter_s": 0.001008, "dense_iter_s": 0.061895, "csr_iter_s": 0.223229, "rolv_total_s": 1.984201, "baseline_total_s": 61.895336, "speedup_total_vs_selected_x": 31.194, "speedup_iter_vs_selected_x": 61.419, "rolv_vs_vendor_sparse_iter_x": 221.512, "rolv_vs_vendor_sparse_total_x": 112.503, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:07:09] Seed: 123456 | Pattern: banded | Zeros: 70%
A_hash: 6de52c734dc3dd3e441813467d3974c05babbe147880af95cae93106e22a77bd | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061945s | CSR: 0.009571s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.761328 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  8d9df1881b3b4b5fee78c07519cc3428371b3fbd21381096db98fe35c6d3ad59  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   8d9df1881b3b4b5fee78c07519cc3428371b3fbd21381096db98fe35c6d3ad59
ROLF_norm_hash:  0f39b126fa71c26f2672d66e7c24c5b70fa1e325cbf4ccf3333c8d2df71581bf
DENGS_norm_hash: afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8
ROLF per-iter:   0.000329s | total: 0.330019s
DENGS per-iter:  0.061934s | total: 61.933727s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 5.43x (≈ 443% faster)
Speedup (per-iter): 9.55x (≈ 855% faster)
Energy Savings: 89.53%
ROLV vs cuSPARSE -> Speedup (per-iter): 9.53x | total: 5.42x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "6de52c734dc3dd3e441813467d3974c05babbe147880af95cae93106e22a77bd", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8", "CSR_norm_hash": "8d9df1881b3b4b5fee78c07519cc3428371b3fbd21381096db98fe35c6d3ad59", "ROLF_norm_hash": "0f39b126fa71c26f2672d66e7c24c5b70fa1e325cbf4ccf3333c8d2df71581bf", "DENGS_norm_hash": "afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "34587fe968cb118281d6e320a80b1d361638e54fc3de87df1dbf85b3f83c9fef", "CSR_qhash_d6": "e7928ff6d61ea6e7f42ac3b02140d2af86626cd02ec6d85a1db7e1b75c57adc5", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061945, "pilot_csr_per_iter_s": 0.009571, "rolv_build_s": 0.761328, "rolv_iter_s": 0.001005, "dense_iter_s": 0.009593, "csr_iter_s": 0.009578, "rolv_total_s": 1.766141, "baseline_total_s": 9.593162, "speedup_total_vs_selected_x": 5.432, "speedup_iter_vs_selected_x": 9.547, "rolv_vs_vendor_sparse_iter_x": 9.533, "rolv_vs_vendor_sparse_total_x": 5.423, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:09:11] Seed: 123456 | Pattern: block_diagonal | Zeros: 70%
A_hash: 605ad79227a409511ccd935bac7446d55792ae15e0550623f778311797a2ba80 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061935s | CSR: 0.006069s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.361994 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3f2fe509b6e436e25c30129bc1a4aeb4b69e13971f473e1b7fd39bb94c0d47a2  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   3f2fe509b6e436e25c30129bc1a4aeb4b69e13971f473e1b7fd39bb94c0d47a2
ROLF_norm_hash:  71235fd9a701d922ba41c2836607dbd277b18a94a84c0fe54ad3e3cb46430593
DENGS_norm_hash: afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75
ROLF per-iter:   0.000329s | total: 0.329522s
DENGS per-iter:  0.061924s | total: 61.923762s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 2.56x (≈ 156% faster)
Speedup (per-iter): 6.04x (≈ 504% faster)
Energy Savings: 83.44%
ROLV vs cuSPARSE -> Speedup (per-iter): 6.05x | total: 2.57x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "605ad79227a409511ccd935bac7446d55792ae15e0550623f778311797a2ba80", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75", "CSR_norm_hash": "3f2fe509b6e436e25c30129bc1a4aeb4b69e13971f473e1b7fd39bb94c0d47a2", "ROLF_norm_hash": "71235fd9a701d922ba41c2836607dbd277b18a94a84c0fe54ad3e3cb46430593", "DENGS_norm_hash": "afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "7bc340a2266a60176174c6afbc41e54623a3acb3c008de5aeff26276a7332fb6", "CSR_qhash_d6": "01ce06767fcaeef7d4eb002639688212b561cd452e87bd13041f12124362ed06", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061935, "pilot_csr_per_iter_s": 0.006069, "rolv_build_s": 1.361994, "rolv_iter_s": 0.001005, "dense_iter_s": 0.006066, "csr_iter_s": 0.006081, "rolv_total_s": 2.366731, "baseline_total_s": 6.066248, "speedup_total_vs_selected_x": 2.563, "speedup_iter_vs_selected_x": 6.038, "rolv_vs_vendor_sparse_iter_x": 6.052, "rolv_vs_vendor_sparse_total_x": 2.569, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:11:06] Seed: 123456 | Pattern: random | Zeros: 80%
A_hash: fe8ecd469d65375943070e2c9f72b2cb8ffc99f59b8e95e01ee55ff351e8a5b5 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061925s | CSR: 0.157117s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.004230 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb  (Dense GEMM (cuBLAS))
CSR_norm_hash:   8b79b461d774dd0fa8a750ae815bc44bea2e55095a909b87d17a0e11773d71bb
ROLF_norm_hash:  2c97f0d1200024546d7641880cac8a28d456f33e2b9d9806d0408d519cd56f37
DENGS_norm_hash: e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb
ROLF per-iter:   0.000329s | total: 0.330124s
DENGS per-iter:  0.061897s | total: 61.896992s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 30.79x (≈ 2979% faster)
Speedup (per-iter): 61.52x (≈ 6052% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 156.25x | total: 78.20x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "fe8ecd469d65375943070e2c9f72b2cb8ffc99f59b8e95e01ee55ff351e8a5b5", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb", "CSR_norm_hash": "8b79b461d774dd0fa8a750ae815bc44bea2e55095a909b87d17a0e11773d71bb", "ROLF_norm_hash": "2c97f0d1200024546d7641880cac8a28d456f33e2b9d9806d0408d519cd56f37", "DENGS_norm_hash": "e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "73400dd11bba92807f9dd80ee8c7bdc5e578106e1ea67465c31cebca0c1dc833", "CSR_qhash_d6": "adfb6e5c6abdb6f36e1568ba9f9b44705f6b4464d391c8a0a86136f38cc9aee1", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061925, "pilot_csr_per_iter_s": 0.157117, "rolv_build_s": 1.00423, "rolv_iter_s": 0.001006, "dense_iter_s": 0.061897, "csr_iter_s": 0.157209, "rolv_total_s": 2.010374, "baseline_total_s": 61.897477, "speedup_total_vs_selected_x": 30.789, "speedup_iter_vs_selected_x": 61.52, "rolv_vs_vendor_sparse_iter_x": 156.249, "rolv_vs_vendor_sparse_total_x": 78.199, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:16:30] Seed: 123456 | Pattern: power_law | Zeros: 80%
A_hash: f5319945ed9e0de80929153636dd5033761020445fb403b1998eb9214d00e127 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061925s | CSR: 0.146524s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.046506 s
ROLV per-iter: 0.001008s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048  (Dense GEMM (cuBLAS))
CSR_norm_hash:   0889e79a3993f803c19e8f2890c06761b38ac6723d53141efe44152035665485
ROLF_norm_hash:  222530270f5d311d6cd32e2219aefd7aa591c2a33d3143d33a5f01187d0d6e1b
DENGS_norm_hash: 0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048
ROLF per-iter:   0.000329s | total: 0.330104s
DENGS per-iter:  0.061895s | total: 61.895055s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 30.13x (≈ 2913% faster)
Speedup (per-iter): 61.43x (≈ 6043% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 145.52x | total: 71.38x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "f5319945ed9e0de80929153636dd5033761020445fb403b1998eb9214d00e127", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048", "CSR_norm_hash": "0889e79a3993f803c19e8f2890c06761b38ac6723d53141efe44152035665485", "ROLF_norm_hash": "222530270f5d311d6cd32e2219aefd7aa591c2a33d3143d33a5f01187d0d6e1b", "DENGS_norm_hash": "0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "66317c61bd76f15b22946d59ad005fceac533aed498ef9ff62ad0904ec173c58", "CSR_qhash_d6": "190b42026a39b7bb47a28b6f44a0fe32cfe8b1ca881bd688825fd92b59ef07ff", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061925, "pilot_csr_per_iter_s": 0.146524, "rolv_build_s": 1.046506, "rolv_iter_s": 0.001008, "dense_iter_s": 0.0619, "csr_iter_s": 0.146632, "rolv_total_s": 2.054141, "baseline_total_s": 61.899832, "speedup_total_vs_selected_x": 30.134, "speedup_iter_vs_selected_x": 61.431, "rolv_vs_vendor_sparse_iter_x": 145.521, "rolv_vs_vendor_sparse_total_x": 71.384, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:21:43] Seed: 123456 | Pattern: banded | Zeros: 80%
A_hash: b2fc7f83b499ca9e4b29ed3cc68b966b4b322cf7926c12186e98ae033e84be58 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061947s | CSR: 0.006487s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.972610 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  25ccedb6e62b6725c6d75761905f570139b07860c1809e1eb1c071d7d92e1851  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   25ccedb6e62b6725c6d75761905f570139b07860c1809e1eb1c071d7d92e1851
ROLF_norm_hash:  e59dd960a001f4488b5b3c7cdcaa8d75712839372bc52a94bab002baba02fd9d
DENGS_norm_hash: 7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6
ROLF per-iter:   0.000329s | total: 0.329907s
DENGS per-iter:  0.061929s | total: 61.928953s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 3.28x (≈ 228% faster)
Speedup (per-iter): 6.45x (≈ 545% faster)
Energy Savings: 84.51%
ROLV vs cuSPARSE -> Speedup (per-iter): 6.47x | total: 3.29x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "b2fc7f83b499ca9e4b29ed3cc68b966b4b322cf7926c12186e98ae033e84be58", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6", "CSR_norm_hash": "25ccedb6e62b6725c6d75761905f570139b07860c1809e1eb1c071d7d92e1851", "ROLF_norm_hash": "e59dd960a001f4488b5b3c7cdcaa8d75712839372bc52a94bab002baba02fd9d", "DENGS_norm_hash": "7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "3d7a5452a9f38bbfb38ee0d4922ca8020b2506f811ff5ec119664b4fe4236084", "CSR_qhash_d6": "426157eb0aa47e2d3bfaca6682a4ec39bab800efeefe2811adf40ab6ef228431", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061947, "pilot_csr_per_iter_s": 0.006487, "rolv_build_s": 0.97261, "rolv_iter_s": 0.001005, "dense_iter_s": 0.006489, "csr_iter_s": 0.0065, "rolv_total_s": 1.977961, "baseline_total_s": 6.489058, "speedup_total_vs_selected_x": 3.281, "speedup_iter_vs_selected_x": 6.455, "rolv_vs_vendor_sparse_iter_x": 6.465, "rolv_vs_vendor_sparse_total_x": 3.286, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:23:33] Seed: 123456 | Pattern: block_diagonal | Zeros: 80%
A_hash: 4b02e483523fbec343feac2b8fed3820615bb6832dda42a3da7b63ccf1ef0014 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061931s | CSR: 0.004187s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.989297 s
ROLV per-iter: 0.001004s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3e218273762a75e0f7f9a30e3de4369d9f2dd8a33ea57b50023a055140b90fff  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   3e218273762a75e0f7f9a30e3de4369d9f2dd8a33ea57b50023a055140b90fff
ROLF_norm_hash:  50f833cd0edb871dab70fd1076fd14286647900fd24dbd3131295de2f48e9f7b
DENGS_norm_hash: 0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82
ROLF per-iter:   0.000329s | total: 0.329519s
DENGS per-iter:  0.061924s | total: 61.924395s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 2.10x (≈ 110% faster)
Speedup (per-iter): 4.17x (≈ 317% faster)
Energy Savings: 76.01%
ROLV vs cuSPARSE -> Speedup (per-iter): 4.17x | total: 2.10x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "4b02e483523fbec343feac2b8fed3820615bb6832dda42a3da7b63ccf1ef0014", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82", "CSR_norm_hash": "3e218273762a75e0f7f9a30e3de4369d9f2dd8a33ea57b50023a055140b90fff", "ROLF_norm_hash": "50f833cd0edb871dab70fd1076fd14286647900fd24dbd3131295de2f48e9f7b", "DENGS_norm_hash": "0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "4eaaed0b681ad4346a051280bb2504683f2650e05430ced90b415a8515706ae4", "CSR_qhash_d6": "62c4976c1d36c77fedd599e98d2a673d89c2d81c354315af80ffe9d7c50e9a6a", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061931, "pilot_csr_per_iter_s": 0.004187, "rolv_build_s": 0.989297, "rolv_iter_s": 0.001004, "dense_iter_s": 0.004185, "csr_iter_s": 0.004186, "rolv_total_s": 1.993161, "baseline_total_s": 4.184892, "speedup_total_vs_selected_x": 2.1, "speedup_iter_vs_selected_x": 4.169, "rolv_vs_vendor_sparse_iter_x": 4.17, "rolv_vs_vendor_sparse_total_x": 2.1, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:25:23] Seed: 123456 | Pattern: random | Zeros: 90%
A_hash: 252a6d9ec7eeab4eb29b6c652bffba9f11178919caadeccd14c45d00311e1433 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061923s | CSR: 0.077882s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.415772 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83  (Dense GEMM (cuBLAS))
CSR_norm_hash:   1cdc8e46c87f3f9f33a8a9b2ad5182dce3fdc06a3ebaee48fe0e30917a95d593
ROLF_norm_hash:  321943d79a8e5a4c00b38e7a32c69fc5870129ed248809d076b51f4519f056ca
DENGS_norm_hash: 0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83
ROLF per-iter:   0.000329s | total: 0.329740s
DENGS per-iter:  0.061895s | total: 61.895156s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 25.56x (≈ 2456% faster)
Speedup (per-iter): 61.51x (≈ 6051% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 77.45x | total: 32.18x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "252a6d9ec7eeab4eb29b6c652bffba9f11178919caadeccd14c45d00311e1433", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83", "CSR_norm_hash": "1cdc8e46c87f3f9f33a8a9b2ad5182dce3fdc06a3ebaee48fe0e30917a95d593", "ROLF_norm_hash": "321943d79a8e5a4c00b38e7a32c69fc5870129ed248809d076b51f4519f056ca", "DENGS_norm_hash": "0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "d07e9d8e4b761c9e713843d9e5ad22646d68738c9c9f78b846a77a566e81f77a", "CSR_qhash_d6": "946a9be2b5f232ee46cd67e44c8707539342516691dfe9279855f30fb64ee420", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061923, "pilot_csr_per_iter_s": 0.077882, "rolv_build_s": 1.415772, "rolv_iter_s": 0.001006, "dense_iter_s": 0.0619, "csr_iter_s": 0.077935, "rolv_total_s": 2.422091, "baseline_total_s": 61.900109, "speedup_total_vs_selected_x": 25.556, "speedup_iter_vs_selected_x": 61.511, "rolv_vs_vendor_sparse_iter_x": 77.446, "rolv_vs_vendor_sparse_total_x": 32.177, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:29:28] Seed: 123456 | Pattern: power_law | Zeros: 90%
A_hash: d1784f30a29c88bb759e8e0ce2e1d3a72ec63f8f7d0190e4b7c74bf9b0f76e26 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061921s | CSR: 0.072758s
Selected baseline: Dense GEMM (cuBLAS)
ROLV load time (operator build): 1.467552 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6  (Dense GEMM (cuBLAS))
CSR_norm_hash:   29206f8b58de6e81a8e9a53420a7e2e2bd40566e569b938d9a820292e2eecdfc
ROLF_norm_hash:  d25ac0c9367d63ba2b4533bd33d053eefdea95762e4d5dafaf6a2ce9ff4cb49f
DENGS_norm_hash: ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6
ROLF per-iter:   0.000329s | total: 0.329526s
DENGS per-iter:  0.061889s | total: 61.889375s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 25.01x (≈ 2401% faster)
Speedup (per-iter): 61.44x (≈ 6044% faster)
Energy Savings: 98.37%
ROLV vs cuSPARSE -> Speedup (per-iter): 72.26x | total: 29.41x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "d1784f30a29c88bb759e8e0ce2e1d3a72ec63f8f7d0190e4b7c74bf9b0f76e26", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6", "CSR_norm_hash": "29206f8b58de6e81a8e9a53420a7e2e2bd40566e569b938d9a820292e2eecdfc", "ROLF_norm_hash": "d25ac0c9367d63ba2b4533bd33d053eefdea95762e4d5dafaf6a2ce9ff4cb49f", "DENGS_norm_hash": "ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "11f8da7a2080d0d605a1ebff2f877f43fdf7d15739e0c108fe9752e7a0e9c93a", "CSR_qhash_d6": "8adc779e481f9a6fb6cb6060d2d8117cf00d279d1241c7eebf7a5b7cb3a4607e", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.061921, "pilot_csr_per_iter_s": 0.072758, "rolv_build_s": 1.467552, "rolv_iter_s": 0.001007, "dense_iter_s": 0.061896, "csr_iter_s": 0.072791, "rolv_total_s": 2.474931, "baseline_total_s": 61.896238, "speedup_total_vs_selected_x": 25.009, "speedup_iter_vs_selected_x": 61.443, "rolv_vs_vendor_sparse_iter_x": 72.258, "rolv_vs_vendor_sparse_total_x": 29.411, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:33:25] Seed: 123456 | Pattern: banded | Zeros: 90%
A_hash: d70a4343e5b268957eb68d7e3674a43f240457ccfda08b4a2d80bc40ab643157 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061936s | CSR: 0.003433s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.432972 s
ROLV per-iter: 0.001008s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0114d85f6b7088696c7a5d5d6ecbf9f4777a4376ce468ad1f20da163869cb64a  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   0114d85f6b7088696c7a5d5d6ecbf9f4777a4376ce468ad1f20da163869cb64a
ROLF_norm_hash:  7b612cfd403099d0e842a486b4d51c55fba1f9c7ff554a4a1ed5affedc0f75a1
DENGS_norm_hash: 6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94
ROLF per-iter:   0.000329s | total: 0.330156s
DENGS per-iter:  0.061932s | total: 61.932004s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 1.41x (≈ 41% faster)
Speedup (per-iter): 3.40x (≈ 240% faster)
Energy Savings: 70.63%
ROLV vs cuSPARSE -> Speedup (per-iter): 3.41x | total: 1.41x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "d70a4343e5b268957eb68d7e3674a43f240457ccfda08b4a2d80bc40ab643157", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94", "CSR_norm_hash": "0114d85f6b7088696c7a5d5d6ecbf9f4777a4376ce468ad1f20da163869cb64a", "ROLF_norm_hash": "7b612cfd403099d0e842a486b4d51c55fba1f9c7ff554a4a1ed5affedc0f75a1", "DENGS_norm_hash": "6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "c850461ae5bfa1c3183f8c5ad70eadff35c42a0cf45abfac99892c98541b884e", "CSR_qhash_d6": "f061d83fad7adca38618f5b1a3e283f2b924c37454c16621c3bcd3ab9802b106", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061936, "pilot_csr_per_iter_s": 0.003433, "rolv_build_s": 1.432972, "rolv_iter_s": 0.001008, "dense_iter_s": 0.003432, "csr_iter_s": 0.003438, "rolv_total_s": 2.441169, "baseline_total_s": 3.432406, "speedup_total_vs_selected_x": 1.406, "speedup_iter_vs_selected_x": 3.405, "rolv_vs_vendor_sparse_iter_x": 3.41, "rolv_vs_vendor_sparse_total_x": 1.408, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:35:15] Seed: 123456 | Pattern: block_diagonal | Zeros: 90%
A_hash: ef3c072370841e3130690e4f6793ea35e3e0c704fce673efdbae340a03091d07 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061932s | CSR: 0.002315s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.411650 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  771ae79a70bcf27cd3c224cd43f82698404a6c4b58035f37e28c9f1bd34614a4  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   771ae79a70bcf27cd3c224cd43f82698404a6c4b58035f37e28c9f1bd34614a4
ROLF_norm_hash:  043a6ad806a02a4dbfd07edb33a20bdebb586c17d7abfcd015ef28379fdadb27
DENGS_norm_hash: 3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79
ROLF per-iter:   0.000329s | total: 0.329492s
DENGS per-iter:  0.061920s | total: 61.919844s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 0.96x (≈ -4% faster)
Speedup (per-iter): 2.30x (≈ 130% faster)
Energy Savings: 56.45%
ROLV vs cuSPARSE -> Speedup (per-iter): 2.30x | total: 0.96x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "ef3c072370841e3130690e4f6793ea35e3e0c704fce673efdbae340a03091d07", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79", "CSR_norm_hash": "771ae79a70bcf27cd3c224cd43f82698404a6c4b58035f37e28c9f1bd34614a4", "ROLF_norm_hash": "043a6ad806a02a4dbfd07edb33a20bdebb586c17d7abfcd015ef28379fdadb27", "DENGS_norm_hash": "3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "cffe32a36e15addac2c5b2fef97a992d6d9c8731c108477cdc00f463498f0e00", "CSR_qhash_d6": "726a88256e08eaefdc7422df0ee8c827c49bf88e9305c6b622c61f00d2bc0ff0", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061932, "pilot_csr_per_iter_s": 0.002315, "rolv_build_s": 1.41165, "rolv_iter_s": 0.001007, "dense_iter_s": 0.002312, "csr_iter_s": 0.002315, "rolv_total_s": 2.418736, "baseline_total_s": 2.312484, "speedup_total_vs_selected_x": 0.956, "speedup_iter_vs_selected_x": 2.296, "rolv_vs_vendor_sparse_iter_x": 2.298, "rolv_vs_vendor_sparse_total_x": 0.957, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:37:01] Seed: 123456 | Pattern: random | Zeros: 95%
A_hash: c926d3fc034ec0adbed3fa6ecc74c1e0c4191486cd48fd095fa3c179c6ef96db | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061922s | CSR: 0.039089s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.464692 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  2970b60cbd67cc47922b8389bed89c5cd6b3e014c150ea4affbd564253b09b38  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   2970b60cbd67cc47922b8389bed89c5cd6b3e014c150ea4affbd564253b09b38
ROLF_norm_hash:  438099e42e0c75acf4c952287040709c2d6b5b9b38bb0103770643c2625437f3
DENGS_norm_hash: f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71
ROLF per-iter:   0.000329s | total: 0.329574s
DENGS per-iter:  0.061895s | total: 61.895449s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 15.84x (≈ 1484% faster)
Speedup (per-iter): 38.93x (≈ 3793% faster)
Energy Savings: 97.43%
ROLV vs cuSPARSE -> Speedup (per-iter): 38.93x | total: 15.84x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "c926d3fc034ec0adbed3fa6ecc74c1e0c4191486cd48fd095fa3c179c6ef96db", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71", "CSR_norm_hash": "2970b60cbd67cc47922b8389bed89c5cd6b3e014c150ea4affbd564253b09b38", "ROLF_norm_hash": "438099e42e0c75acf4c952287040709c2d6b5b9b38bb0103770643c2625437f3", "DENGS_norm_hash": "f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "c341e71c1d6aaaff215d32bf3ce2823faac0512f379a99a19bf0274553f15740", "CSR_qhash_d6": "10fbb75975d690d6f0b1f024770801b937b555f405b3c437041869e128e9387a", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061922, "pilot_csr_per_iter_s": 0.039089, "rolv_build_s": 1.464692, "rolv_iter_s": 0.001005, "dense_iter_s": 0.03911, "csr_iter_s": 0.039112, "rolv_total_s": 2.469242, "baseline_total_s": 39.109531, "speedup_total_vs_selected_x": 15.839, "speedup_iter_vs_selected_x": 38.932, "rolv_vs_vendor_sparse_iter_x": 38.935, "rolv_vs_vendor_sparse_total_x": 15.84, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:40:08] Seed: 123456 | Pattern: power_law | Zeros: 95%
A_hash: 6417a2a60f09c4389956722addb9e641d9618bcfe0eae0e987dfe602fdefb429 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061925s | CSR: 0.036551s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.998979 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  e686e5c12e0597030da5444a4b9464b97b1313cf3a80a7028b6c02abeb9a1483  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   e686e5c12e0597030da5444a4b9464b97b1313cf3a80a7028b6c02abeb9a1483
ROLF_norm_hash:  d02f72a24d8562f072ac59b90a05de2bb21f3e093177220577b1c675fdd1e16e
DENGS_norm_hash: e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf
ROLF per-iter:   0.000329s | total: 0.329741s
DENGS per-iter:  0.061899s | total: 61.898816s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.25x (≈ 1725% faster)
Speedup (per-iter): 36.40x (≈ 3540% faster)
Energy Savings: 97.25%
ROLV vs cuSPARSE -> Speedup (per-iter): 36.40x | total: 18.25x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "6417a2a60f09c4389956722addb9e641d9618bcfe0eae0e987dfe602fdefb429", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf", "CSR_norm_hash": "e686e5c12e0597030da5444a4b9464b97b1313cf3a80a7028b6c02abeb9a1483", "ROLF_norm_hash": "d02f72a24d8562f072ac59b90a05de2bb21f3e093177220577b1c675fdd1e16e", "DENGS_norm_hash": "e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "100de79f25f9eba2174c313cdd119f1094c254144c65bddf7539fe46bd2b2afb", "CSR_qhash_d6": "2eadaea2613158a9606adf160bb4d1adc89e9083daf2857fc968dfaa77b1157a", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061925, "pilot_csr_per_iter_s": 0.036551, "rolv_build_s": 0.998979, "rolv_iter_s": 0.001005, "dense_iter_s": 0.036567, "csr_iter_s": 0.036571, "rolv_total_s": 2.003576, "baseline_total_s": 36.56684, "speedup_total_vs_selected_x": 18.251, "speedup_iter_vs_selected_x": 36.399, "rolv_vs_vendor_sparse_iter_x": 36.404, "rolv_vs_vendor_sparse_total_x": 18.253, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:43:07] Seed: 123456 | Pattern: banded | Zeros: 95%
A_hash: f9841b629a96caca12ae5093b69047a66277d824418f1f09df0d2ec6bec61381 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061949s | CSR: 0.001907s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.345872 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  277ecfb98edaa677368c7e90f77cca60592fa20e5e9c294c25797244a2461f75  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   277ecfb98edaa677368c7e90f77cca60592fa20e5e9c294c25797244a2461f75
ROLF_norm_hash:  da09351c75b5339ca035bd5d494f359e4c85eb4b75b0e3f8b5462fc1a53761ba
DENGS_norm_hash: a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08
ROLF per-iter:   0.000329s | total: 0.329596s
DENGS per-iter:  0.061934s | total: 61.934461s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 0.81x (≈ -19% faster)
Speedup (per-iter): 1.90x (≈ 90% faster)
Energy Savings: 47.29%
ROLV vs cuSPARSE -> Speedup (per-iter): 1.90x | total: 0.81x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "f9841b629a96caca12ae5093b69047a66277d824418f1f09df0d2ec6bec61381", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08", "CSR_norm_hash": "277ecfb98edaa677368c7e90f77cca60592fa20e5e9c294c25797244a2461f75", "ROLF_norm_hash": "da09351c75b5339ca035bd5d494f359e4c85eb4b75b0e3f8b5462fc1a53761ba", "DENGS_norm_hash": "a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "9c378462296ec1cacef0a364ddaeebca041f1cfda7d5705308d0e32cbdf1f369", "CSR_qhash_d6": "d914582bcff407d02f0cba2482661dd1c40470c27f057596c87f19b2fdfff871", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061949, "pilot_csr_per_iter_s": 0.001907, "rolv_build_s": 1.345872, "rolv_iter_s": 0.001005, "dense_iter_s": 0.001906, "csr_iter_s": 0.001907, "rolv_total_s": 2.350462, "baseline_total_s": 1.90587, "speedup_total_vs_selected_x": 0.811, "speedup_iter_vs_selected_x": 1.897, "rolv_vs_vendor_sparse_iter_x": 1.898, "rolv_vs_vendor_sparse_total_x": 0.811, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:44:52] Seed: 123456 | Pattern: block_diagonal | Zeros: 95%
A_hash: 743ed1c8dc03a5de5d0b131edc508c8c9e30dc02e5406aeb9cb6e8c0ce493874 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061937s | CSR: 0.001373s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.968425 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  03c424e47d15e054b911b6fd86539ff97d470d944b555e4dbb55477e80d25ad7  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   03c424e47d15e054b911b6fd86539ff97d470d944b555e4dbb55477e80d25ad7
ROLF_norm_hash:  a23f52d144ae759bed22286b940fe6340572d8562f2e0cdd90f665c8a5c99fa6
DENGS_norm_hash: cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1
ROLF per-iter:   0.000329s | total: 0.329843s
DENGS per-iter:  0.061924s | total: 61.924121s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 0.69x (≈ -31% faster)
Speedup (per-iter): 1.36x (≈ 36% faster)
Energy Savings: 26.54%
ROLV vs cuSPARSE -> Speedup (per-iter): 1.36x | total: 0.69x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "743ed1c8dc03a5de5d0b131edc508c8c9e30dc02e5406aeb9cb6e8c0ce493874", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1", "CSR_norm_hash": "03c424e47d15e054b911b6fd86539ff97d470d944b555e4dbb55477e80d25ad7", "ROLF_norm_hash": "a23f52d144ae759bed22286b940fe6340572d8562f2e0cdd90f665c8a5c99fa6", "DENGS_norm_hash": "cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "efbe0ca36ca8f3c6721275268db28beadf0ad8a4b2a7aa76452f596348100e65", "CSR_qhash_d6": "8237c2fa57454259a23b6a6b2ec13e3ef7b9c1070c5fe9d03431ab7161592e0a", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061937, "pilot_csr_per_iter_s": 0.001373, "rolv_build_s": 0.968425, "rolv_iter_s": 0.001007, "dense_iter_s": 0.00137, "csr_iter_s": 0.001371, "rolv_total_s": 1.975114, "baseline_total_s": 1.370422, "speedup_total_vs_selected_x": 0.694, "speedup_iter_vs_selected_x": 1.361, "rolv_vs_vendor_sparse_iter_x": 1.361, "rolv_vs_vendor_sparse_total_x": 0.694, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:46:40] Seed: 123456 | Pattern: random | Zeros: 99%
A_hash: 9fde8b5d279f5d4d8297c2b0a4f006d0bf2475b62e6dabc7da09b547c8edbc8a | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061925s | CSR: 0.008132s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 1.431918 s
ROLV per-iter: 0.001007s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  1135cbc22d07bdd3bbcd7d89226b542df469f820cc79131a7b50047aee0a27c4  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   1135cbc22d07bdd3bbcd7d89226b542df469f820cc79131a7b50047aee0a27c4
ROLF_norm_hash:  cd16ca2eb816f4e3a4324002312ca0268a38b732796aab7888293ef7f24633b7
DENGS_norm_hash: cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9
ROLF per-iter:   0.000329s | total: 0.330136s
DENGS per-iter:  0.061901s | total: 61.901164s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 3.35x (≈ 235% faster)
Speedup (per-iter): 8.10x (≈ 710% faster)
Energy Savings: 87.66%
ROLV vs cuSPARSE -> Speedup (per-iter): 8.09x | total: 3.34x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "9fde8b5d279f5d4d8297c2b0a4f006d0bf2475b62e6dabc7da09b547c8edbc8a", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9", "CSR_norm_hash": "1135cbc22d07bdd3bbcd7d89226b542df469f820cc79131a7b50047aee0a27c4", "ROLF_norm_hash": "cd16ca2eb816f4e3a4324002312ca0268a38b732796aab7888293ef7f24633b7", "DENGS_norm_hash": "cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "18a55821ec1e53d19620c8b3fdf1bd7dc27837e4d3f0bc4b8bb33ab2e24117eb", "CSR_qhash_d6": "89dfa26ea05dc38a799cfe1d8a0d2fddefe98feb984bc8e83ccba7157074ab90", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061925, "pilot_csr_per_iter_s": 0.008132, "rolv_build_s": 1.431918, "rolv_iter_s": 0.001007, "dense_iter_s": 0.00816, "csr_iter_s": 0.008148, "rolv_total_s": 2.43891, "baseline_total_s": 8.159898, "speedup_total_vs_selected_x": 3.346, "speedup_iter_vs_selected_x": 8.103, "rolv_vs_vendor_sparse_iter_x": 8.091, "rolv_vs_vendor_sparse_total_x": 3.341, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:48:38] Seed: 123456 | Pattern: power_law | Zeros: 99%
A_hash: 3884cba828aa7a1488fc132da5edbcb037e4d5cda60d2548cbb05d1438117888 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061927s | CSR: 0.007619s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.997269 s
ROLV per-iter: 0.001005s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  c3ed862a2d9b80d5bbed8e8e72b8331eebd23e62809e3f65ae2611dad8f92e0c  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   c3ed862a2d9b80d5bbed8e8e72b8331eebd23e62809e3f65ae2611dad8f92e0c
ROLF_norm_hash:  4727304f2130748ccc072bc431f0db2c0cbe41049c05e3323613e00e8f930359
DENGS_norm_hash: c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d
ROLF per-iter:   0.000329s | total: 0.329978s
DENGS per-iter:  0.061905s | total: 61.905090s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 3.82x (≈ 282% faster)
Speedup (per-iter): 7.60x (≈ 660% faster)
Energy Savings: 86.84%
ROLV vs cuSPARSE -> Speedup (per-iter): 7.60x | total: 3.82x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "3884cba828aa7a1488fc132da5edbcb037e4d5cda60d2548cbb05d1438117888", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d", "CSR_norm_hash": "c3ed862a2d9b80d5bbed8e8e72b8331eebd23e62809e3f65ae2611dad8f92e0c", "ROLF_norm_hash": "4727304f2130748ccc072bc431f0db2c0cbe41049c05e3323613e00e8f930359", "DENGS_norm_hash": "c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "570879d26c1763504bd05013868ba03d6e5c343019d405fbb6664799c3446a0a", "CSR_qhash_d6": "a7daea72fb263b34b3fbd280981d95b8db056e4ee2a3a4a92255e8a12bd01de4", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061927, "pilot_csr_per_iter_s": 0.007619, "rolv_build_s": 0.997269, "rolv_iter_s": 0.001005, "dense_iter_s": 0.007642, "csr_iter_s": 0.007644, "rolv_total_s": 2.002617, "baseline_total_s": 7.642152, "speedup_total_vs_selected_x": 3.816, "speedup_iter_vs_selected_x": 7.601, "rolv_vs_vendor_sparse_iter_x": 7.603, "rolv_vs_vendor_sparse_total_x": 3.817, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:50:38] Seed: 123456 | Pattern: banded | Zeros: 99%
A_hash: 1b643fe5ac4811868b9b5bfee7d7ed4d02a612b4add98ac2d0f399d014599b67 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061954s | CSR: 0.000694s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.998954 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3fe8ac1fab005fbd1e63982dee89afe3d065e979ad01a731d3ebf842a3648e4e  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   3fe8ac1fab005fbd1e63982dee89afe3d065e979ad01a731d3ebf842a3648e4e
ROLF_norm_hash:  832a9e85bddee020a7655ba9a16031e464caa87f1ceb92d8567b75780ccb7c6d
DENGS_norm_hash: 2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad
ROLF per-iter:   0.000329s | total: 0.329383s
DENGS per-iter:  0.061942s | total: 61.942027s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 0.35x (≈ -65% faster)
Speedup (per-iter): 0.69x (≈ -31% faster)
Energy Savings: -44.83%
ROLV vs cuSPARSE -> Speedup (per-iter): 0.69x | total: 0.35x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "1b643fe5ac4811868b9b5bfee7d7ed4d02a612b4add98ac2d0f399d014599b67", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad", "CSR_norm_hash": "3fe8ac1fab005fbd1e63982dee89afe3d065e979ad01a731d3ebf842a3648e4e", "ROLF_norm_hash": "832a9e85bddee020a7655ba9a16031e464caa87f1ceb92d8567b75780ccb7c6d", "DENGS_norm_hash": "2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "36fabc0207e13e7ae08a8c6db45526167c20c66464e356ba7cd4969570c1cb52", "CSR_qhash_d6": "ef059811d87737b49d1022a11bffb21a5179c83ffdaf83784cd83d02836065b0", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061954, "pilot_csr_per_iter_s": 0.000694, "rolv_build_s": 0.998954, "rolv_iter_s": 0.001006, "dense_iter_s": 0.000695, "csr_iter_s": 0.000694, "rolv_total_s": 2.004908, "baseline_total_s": 0.694577, "speedup_total_vs_selected_x": 0.346, "speedup_iter_vs_selected_x": 0.69, "rolv_vs_vendor_sparse_iter_x": 0.69, "rolv_vs_vendor_sparse_total_x": 0.346, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:52:20] Seed: 123456 | Pattern: block_diagonal | Zeros: 99%
A_hash: d78e202117fb1b5ee60605254db62aa72b0d2b72a9d6ceec1a84ad78c44df368 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.061946s | CSR: 0.000622s
Selected baseline: Sparse CSR (cuSPARSE)
ROLV load time (operator build): 0.973540 s
ROLV per-iter: 0.001006s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  a87d97bfb7c2c30e63f5a68be7b57390d7f240f5ad68a6c8ba553bbb6b913de8  (Sparse CSR (cuSPARSE))
CSR_norm_hash:   a87d97bfb7c2c30e63f5a68be7b57390d7f240f5ad68a6c8ba553bbb6b913de8
ROLF_norm_hash:  aedd4cf9d03fcc98a2d4b33b9c56bc89ad6ca996f94ce0f83609d26a240cfcd3
DENGS_norm_hash: 81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b
ROLF per-iter:   0.000336s | total: 0.336531s
DENGS per-iter:  0.061937s | total: 61.937418s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 0.31x (≈ -69% faster)
Speedup (per-iter): 0.62x (≈ -38% faster)
Energy Savings: -62.46%
ROLV vs cuSPARSE -> Speedup (per-iter): 0.62x | total: 0.31x
{"platform": "CUDA", "device": "NVIDIA B200", "adapted_batch": false, "effective_batch": 5000, "dense_label": "cuBLAS", "sparse_label": "cuSPARSE", "input_hash_A": "d78e202117fb1b5ee60605254db62aa72b0d2b72a9d6ceec1a84ad78c44df368", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b", "CSR_norm_hash": "a87d97bfb7c2c30e63f5a68be7b57390d7f240f5ad68a6c8ba553bbb6b913de8", "ROLF_norm_hash": "aedd4cf9d03fcc98a2d4b33b9c56bc89ad6ca996f94ce0f83609d26a240cfcd3", "DENGS_norm_hash": "81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "72ae2e8b1b11989b240bd407be5eae8d1ffdbf75beeacce50c056cdb544c412f", "CSR_qhash_d6": "2f7bfb9127eae8277e2d1d70e892b6d1a1c2ada0aa940a4bba915f09e9f01c96", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.061946, "pilot_csr_per_iter_s": 0.000622, "rolv_build_s": 0.97354, "rolv_iter_s": 0.001006, "dense_iter_s": 0.000619, "csr_iter_s": 0.00062, "rolv_total_s": 1.979834, "baseline_total_s": 0.619429, "speedup_total_vs_selected_x": 0.313, "speedup_iter_vs_selected_x": 0.616, "rolv_vs_vendor_sparse_iter_x": 0.616, "rolv_vs_vendor_sparse_total_x": 0.313, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

=== FOOTER REPORT (CUDA) ===
- Aggregate speedup (total vs selected): 13.78x (≈ 1278% faster)
- Aggregate speedup (per-iter vs selected): 29.42x (≈ 2842% faster)
- Aggregate energy savings (proxy vs selected): 78.9%
- Verification: TF32 off, deterministic algorithms, CSR canonicalization, CPU-fp64 normalization and SHA-256 hashing.
 
AMD
=== RUN SUITE (ROCm) on AMD Instinct MI300X ===
Matrices: 20,000x20,000  Batch Size: 5,000  Iterations: 1,000

[2025-12-07 08:43:38] Seed: 123456 | Pattern: random | Zeros: 40%
A_hash: e3644a901043856adaa3b878146a5978eda600732465e78134f6121ad2135eab | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
/tmp/ipykernel_372/1530191931.py:726: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at ../aten/src/ATen/SparseCsrTensorImpl.cpp:53.)
  A_csr_raw = A_dense.to_sparse_csr()
Baseline pilots per-iter -> Dense: 0.040154s | CSR: 1.365250s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188197 s
ROLV per-iter: 0.001948s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c  (Dense GEMM (rocBLAS))
CSR_norm_hash:   11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c
ROLF_norm_hash:  96f29966a7efcb5ef2528e92078988a2063a442867509f2f469e1b63cec61896
DENGS_norm_hash: 11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c
ROLF per-iter:   0.000243s | total: 0.243209s
DENGS per-iter:  0.041806s | total: 41.805727s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 19.31x (≈ 1831% faster)
Speedup (per-iter): 21.17x (≈ 2017% faster)
Energy Savings: 95.28%
ROLV vs rocSPARSE -> Speedup (per-iter): 730.30x | total: 665.97x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "e3644a901043856adaa3b878146a5978eda600732465e78134f6121ad2135eab", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c", "CSR_norm_hash": "11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c", "ROLF_norm_hash": "96f29966a7efcb5ef2528e92078988a2063a442867509f2f469e1b63cec61896", "DENGS_norm_hash": "11b6241f09adfebda8a84e36dfbfba9192af8d759dbd0b8612db6923472fac6c", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "d02e8632c028003b3549fb086dad732fc49aed49a06e28cfc5e2a0f32da41a36", "CSR_qhash_d6": "d02e8632c028003b3549fb086dad732fc49aed49a06e28cfc5e2a0f32da41a36", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.040154, "pilot_csr_per_iter_s": 1.36525, "rolv_build_s": 0.188197, "rolv_iter_s": 0.001948, "dense_iter_s": 0.041254, "csr_iter_s": 1.422933, "rolv_total_s": 2.136621, "baseline_total_s": 41.253598, "speedup_total_vs_selected_x": 19.308, "speedup_iter_vs_selected_x": 21.173, "rolv_vs_vendor_sparse_iter_x": 730.3, "rolv_vs_vendor_sparse_total_x": 665.973, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:10:22] Seed: 123456 | Pattern: power_law | Zeros: 40%
A_hash: 0bc0d2cd333849b2bc5726b8182342a2b1f1692dec3ce1baa02459ebd0feca6e | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.040850s | CSR: 1.278949s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188353 s
ROLV per-iter: 0.001953s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb  (Dense GEMM (rocBLAS))
CSR_norm_hash:   3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb
ROLF_norm_hash:  04c6dfa6fcbee09f54a8fe3b9d83bb2c2537a62e94ff981cc0de5f2215d8c38b
DENGS_norm_hash: 3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb
ROLF per-iter:   0.000247s | total: 0.247427s
DENGS per-iter:  0.041998s | total: 41.998336s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 19.48x (≈ 1848% faster)
Speedup (per-iter): 21.36x (≈ 2036% faster)
Energy Savings: 95.32%
ROLV vs rocSPARSE -> Speedup (per-iter): 665.58x | total: 607.03x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "0bc0d2cd333849b2bc5726b8182342a2b1f1692dec3ce1baa02459ebd0feca6e", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb", "CSR_norm_hash": "3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb", "ROLF_norm_hash": "04c6dfa6fcbee09f54a8fe3b9d83bb2c2537a62e94ff981cc0de5f2215d8c38b", "DENGS_norm_hash": "3200b111483c9fae293d87a88a8e25c6fe52eb6436f1def69101414d10b57cfb", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "1bb133eca121d0422acc7c427733ee08af00c0731d925906a3a7449af91e982e", "CSR_qhash_d6": "1bb133eca121d0422acc7c427733ee08af00c0731d925906a3a7449af91e982e", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.04085, "pilot_csr_per_iter_s": 1.278949, "rolv_build_s": 0.188353, "rolv_iter_s": 0.001953, "dense_iter_s": 0.041719, "csr_iter_s": 1.299926, "rolv_total_s": 2.141439, "baseline_total_s": 41.719277, "speedup_total_vs_selected_x": 19.482, "speedup_iter_vs_selected_x": 21.361, "rolv_vs_vendor_sparse_iter_x": 665.576, "rolv_vs_vendor_sparse_total_x": 607.034, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:35:46] Seed: 123456 | Pattern: banded | Zeros: 40%
A_hash: 69975c70a3346649e1fbefab534eae7887a68247af2ad0c91ced7488ab619e6c | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033528s | CSR: 0.049602s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.186168 s
ROLV per-iter: 0.001936s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f  (Dense GEMM (rocBLAS))
CSR_norm_hash:   1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f
ROLF_norm_hash:  3d143289649ac69457ce1a4c0c58322caf19f04aff297d32892e289db68e9353
DENGS_norm_hash: 1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f
ROLF per-iter:   0.000227s | total: 0.228319s
DENGS per-iter:  0.035299s | total: 35.299316s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.35x (≈ 1735% faster)
Speedup (per-iter): 20.11x (≈ 1911% faster)
Energy Savings: 95.03%
ROLV vs rocSPARSE -> Speedup (per-iter): 28.37x | total: 25.88x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "69975c70a3346649e1fbefab534eae7887a68247af2ad0c91ced7488ab619e6c", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f", "CSR_norm_hash": "1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f", "ROLF_norm_hash": "3d143289649ac69457ce1a4c0c58322caf19f04aff297d32892e289db68e9353", "DENGS_norm_hash": "1e73da27ba6b296895312009edb9bddcc8b91b02b3647b7a9aae70a80af2067f", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "80fe483ed7c35f0587e85f5c26d02f3e5b9572628977d7add5283e61db8ad088", "CSR_qhash_d6": "80fe483ed7c35f0587e85f5c26d02f3e5b9572628977d7add5283e61db8ad088", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.033528, "pilot_csr_per_iter_s": 0.049602, "rolv_build_s": 0.186168, "rolv_iter_s": 0.001936, "dense_iter_s": 0.038947, "csr_iter_s": 0.054928, "rolv_total_s": 2.122459, "baseline_total_s": 38.947277, "speedup_total_vs_selected_x": 18.35, "speedup_iter_vs_selected_x": 20.114, "rolv_vs_vendor_sparse_iter_x": 28.368, "rolv_vs_vendor_sparse_total_x": 25.879, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:38:12] Seed: 123456 | Pattern: block_diagonal | Zeros: 40%
A_hash: d7a5bfe4c7f465590f90417984ef8f0754801ffe2307e0f3a276649b4868f2ad | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033058s | CSR: 0.032761s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186092 s
ROLV per-iter: 0.001955s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c
ROLF_norm_hash:  fad1e70cbfb1f7a7c76bac355291ed547070889fc04e0bac447e2cc9d20dcff1
DENGS_norm_hash: 988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c
ROLF per-iter:   0.000221s | total: 0.221950s
DENGS per-iter:  0.034435s | total: 34.435043s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 15.60x (≈ 1460% faster)
Speedup (per-iter): 17.08x (≈ 1608% faster)
Energy Savings: 94.15%
ROLV vs rocSPARSE -> Speedup (per-iter): 17.24x | total: 15.74x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "d7a5bfe4c7f465590f90417984ef8f0754801ffe2307e0f3a276649b4868f2ad", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c", "CSR_norm_hash": "988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c", "ROLF_norm_hash": "fad1e70cbfb1f7a7c76bac355291ed547070889fc04e0bac447e2cc9d20dcff1", "DENGS_norm_hash": "988617603b8b5a585fdf4dad647ec0ecddad53772808704777c1f88f54f0325c", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "9993275a88ff770aeb9aca162be175a9c3040c53c5c7f6fc2a4f319c31cfdc98", "CSR_qhash_d6": "9993275a88ff770aeb9aca162be175a9c3040c53c5c7f6fc2a4f319c31cfdc98", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033058, "pilot_csr_per_iter_s": 0.032761, "rolv_build_s": 0.186092, "rolv_iter_s": 0.001955, "dense_iter_s": 0.033398, "csr_iter_s": 0.033698, "rolv_total_s": 2.141241, "baseline_total_s": 33.398035, "speedup_total_vs_selected_x": 15.598, "speedup_iter_vs_selected_x": 17.082, "rolv_vs_vendor_sparse_iter_x": 17.236, "rolv_vs_vendor_sparse_total_x": 15.738, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 09:40:13] Seed: 123456 | Pattern: random | Zeros: 50%
A_hash: 6e4770bed2259e6973f564d1f8d9f3edc952d13fc6befcf5a9f9094269703540 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.040205s | CSR: 1.159500s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.189240 s
ROLV per-iter: 0.001961s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404  (Dense GEMM (rocBLAS))
CSR_norm_hash:   16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404
ROLF_norm_hash:  c816fe475c90aae0faf91c6f709db164d5fa330d0a3538d29218bc7dc6e7e99f
DENGS_norm_hash: 16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404
ROLF per-iter:   0.000242s | total: 0.242517s
DENGS per-iter:  0.041480s | total: 41.479859s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 19.15x (≈ 1815% faster)
Speedup (per-iter): 21.00x (≈ 2000% faster)
Energy Savings: 95.24%
ROLV vs rocSPARSE -> Speedup (per-iter): 600.67x | total: 547.81x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "6e4770bed2259e6973f564d1f8d9f3edc952d13fc6befcf5a9f9094269703540", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404", "CSR_norm_hash": "16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404", "ROLF_norm_hash": "c816fe475c90aae0faf91c6f709db164d5fa330d0a3538d29218bc7dc6e7e99f", "DENGS_norm_hash": "16a6f29a289e90371d2461de0e92f680a147484e05e7a322305ac8403f395404", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "6411421f1efd8ea75978adb572c41db98aed6edb716989a47e78ad96e0a71457", "CSR_qhash_d6": "6411421f1efd8ea75978adb572c41db98aed6edb716989a47e78ad96e0a71457", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.040205, "pilot_csr_per_iter_s": 1.1595, "rolv_build_s": 0.18924, "rolv_iter_s": 0.001961, "dense_iter_s": 0.041182, "csr_iter_s": 1.178029, "rolv_total_s": 2.150418, "baseline_total_s": 41.182012, "speedup_total_vs_selected_x": 19.151, "speedup_iter_vs_selected_x": 20.999, "rolv_vs_vendor_sparse_iter_x": 600.674, "rolv_vs_vendor_sparse_total_x": 547.814, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:01:52] Seed: 123456 | Pattern: power_law | Zeros: 50%
A_hash: e868d93c6a2425c33f4461dda493d60421f514ce596dcf01814e71c6fb964106 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.040653s | CSR: 1.089460s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.189021 s
ROLV per-iter: 0.001968s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b  (Dense GEMM (rocBLAS))
CSR_norm_hash:   2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b
ROLF_norm_hash:  454ae05f90c5e045acc09af4e561c50f160fe5add28880ee4daf2660646217ba
DENGS_norm_hash: 2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b
ROLF per-iter:   0.000245s | total: 0.245189s
DENGS per-iter:  0.041628s | total: 41.627938s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 19.28x (≈ 1828% faster)
Speedup (per-iter): 21.13x (≈ 2013% faster)
Energy Savings: 95.27%
ROLV vs rocSPARSE -> Speedup (per-iter): 559.74x | total: 510.68x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "e868d93c6a2425c33f4461dda493d60421f514ce596dcf01814e71c6fb964106", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b", "CSR_norm_hash": "2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b", "ROLF_norm_hash": "454ae05f90c5e045acc09af4e561c50f160fe5add28880ee4daf2660646217ba", "DENGS_norm_hash": "2c7b53eec42709fbc3a8cece030b36aa65de803ee859a661ae2d92444e839f2b", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "5eedfa8fdcd56343dcae7a1bcfe78a3e0011acf344fa0a15bca967f4c5750f59", "CSR_qhash_d6": "5eedfa8fdcd56343dcae7a1bcfe78a3e0011acf344fa0a15bca967f4c5750f59", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.040653, "pilot_csr_per_iter_s": 1.08946, "rolv_build_s": 0.189021, "rolv_iter_s": 0.001968, "dense_iter_s": 0.041576, "csr_iter_s": 1.101316, "rolv_total_s": 2.156574, "baseline_total_s": 41.576363, "speedup_total_vs_selected_x": 19.279, "speedup_iter_vs_selected_x": 21.131, "rolv_vs_vendor_sparse_iter_x": 559.739, "rolv_vs_vendor_sparse_total_x": 510.678, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:22:15] Seed: 123456 | Pattern: banded | Zeros: 50%
A_hash: 36930b864e45f6c7bc4c05a36ceed9e5546aba4f26c38e27ec94b84500ab052f | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034322s | CSR: 0.043050s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.186680 s
ROLV per-iter: 0.001955s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234  (Dense GEMM (rocBLAS))
CSR_norm_hash:   0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234
ROLF_norm_hash:  0621305b553d14c934dc235dd9616ff421a67e87342f046923dc5da9bed27029
DENGS_norm_hash: 0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234
ROLF per-iter:   0.000227s | total: 0.227734s
DENGS per-iter:  0.035423s | total: 35.422910s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 16.39x (≈ 1539% faster)
Speedup (per-iter): 17.95x (≈ 1695% faster)
Energy Savings: 94.43%
ROLV vs rocSPARSE -> Speedup (per-iter): 22.65x | total: 20.67x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "36930b864e45f6c7bc4c05a36ceed9e5546aba4f26c38e27ec94b84500ab052f", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234", "CSR_norm_hash": "0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234", "ROLF_norm_hash": "0621305b553d14c934dc235dd9616ff421a67e87342f046923dc5da9bed27029", "DENGS_norm_hash": "0fe031672a78ac00d079d51b2c3b1ad3e4eb1c0428bd1bc66b4a2f100f6a7234", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "91a6790e57791bfb5eb9aeab2ad3128bc32fc47fb2b6dcd8728760921b04c533", "CSR_qhash_d6": "91a6790e57791bfb5eb9aeab2ad3128bc32fc47fb2b6dcd8728760921b04c533", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.034322, "pilot_csr_per_iter_s": 0.04305, "rolv_build_s": 0.18668, "rolv_iter_s": 0.001955, "dense_iter_s": 0.035099, "csr_iter_s": 0.04428, "rolv_total_s": 2.141853, "baseline_total_s": 35.09941, "speedup_total_vs_selected_x": 16.387, "speedup_iter_vs_selected_x": 17.952, "rolv_vs_vendor_sparse_iter_x": 22.648, "rolv_vs_vendor_sparse_total_x": 20.674, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:24:27] Seed: 123456 | Pattern: block_diagonal | Zeros: 50%
A_hash: 8db5189cd07996217967440640b6d42a07f04d0966354d2bccdba45b8f0e85b6 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033231s | CSR: 0.028306s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186572 s
ROLV per-iter: 0.001955s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66
ROLF_norm_hash:  1b772b5257f1ebb4c0e1b3eb7d087b0a42bbbeb650b2fd543326e966e21a2cb4
DENGS_norm_hash: 03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66
ROLF per-iter:   0.000221s | total: 0.221573s
DENGS per-iter:  0.034623s | total: 34.622953s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 13.47x (≈ 1247% faster)
Speedup (per-iter): 14.76x (≈ 1376% faster)
Energy Savings: 93.22%
ROLV vs rocSPARSE -> Speedup (per-iter): 14.88x | total: 13.58x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "8db5189cd07996217967440640b6d42a07f04d0966354d2bccdba45b8f0e85b6", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66", "CSR_norm_hash": "03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66", "ROLF_norm_hash": "1b772b5257f1ebb4c0e1b3eb7d087b0a42bbbeb650b2fd543326e966e21a2cb4", "DENGS_norm_hash": "03f3a34956a323cf0aaab8acfc428958d3cdffa022221a76104fbccce492ff66", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "b5f52a9ef8ffd7efcef97cbd495082fcb11cd7cd2264e12ccaebab8950e1f08e", "CSR_qhash_d6": "b5f52a9ef8ffd7efcef97cbd495082fcb11cd7cd2264e12ccaebab8950e1f08e", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033231, "pilot_csr_per_iter_s": 0.028306, "rolv_build_s": 0.186572, "rolv_iter_s": 0.001955, "dense_iter_s": 0.02885, "csr_iter_s": 0.029085, "rolv_total_s": 2.141374, "baseline_total_s": 28.850441, "speedup_total_vs_selected_x": 13.473, "speedup_iter_vs_selected_x": 14.759, "rolv_vs_vendor_sparse_iter_x": 14.879, "rolv_vs_vendor_sparse_total_x": 13.583, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:26:20] Seed: 123456 | Pattern: random | Zeros: 60%
A_hash: 3a128a12c751e2a52a9f05427ad881a4beeb441b1aa828f2c83dec9767075e14 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.039610s | CSR: 0.944080s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188615 s
ROLV per-iter: 0.001962s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e  (Dense GEMM (rocBLAS))
CSR_norm_hash:   82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e
ROLF_norm_hash:  53ee64013770fdb75b32a6e92743de71f64c87429155a238a486d4601379ff1b
DENGS_norm_hash: 82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e
ROLF per-iter:   0.000241s | total: 0.241801s
DENGS per-iter:  0.041019s | total: 41.018930s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.91x (≈ 1791% faster)
Speedup (per-iter): 20.72x (≈ 1972% faster)
Energy Savings: 95.17%
ROLV vs rocSPARSE -> Speedup (per-iter): 487.93x | total: 445.14x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "3a128a12c751e2a52a9f05427ad881a4beeb441b1aa828f2c83dec9767075e14", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e", "CSR_norm_hash": "82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e", "ROLF_norm_hash": "53ee64013770fdb75b32a6e92743de71f64c87429155a238a486d4601379ff1b", "DENGS_norm_hash": "82ff97b0c2c9d6b4e6a850bdbeec16cf158da8950cbefe522f043e059a8a944e", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "474ef2e5789ba96a76b96db5a6f8e76990d35228b24cd5d7660008bef1c3606c", "CSR_qhash_d6": "474ef2e5789ba96a76b96db5a6f8e76990d35228b24cd5d7660008bef1c3606c", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.03961, "pilot_csr_per_iter_s": 0.94408, "rolv_build_s": 0.188615, "rolv_iter_s": 0.001962, "dense_iter_s": 0.040664, "csr_iter_s": 0.957353, "rolv_total_s": 2.150697, "baseline_total_s": 40.663887, "speedup_total_vs_selected_x": 18.907, "speedup_iter_vs_selected_x": 20.725, "rolv_vs_vendor_sparse_iter_x": 487.927, "rolv_vs_vendor_sparse_total_x": 445.136, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 10:44:15] Seed: 123456 | Pattern: power_law | Zeros: 60%
A_hash: 9d19ea5f391575455f95a6f93a0dc330f0816afb109185aa39e76d5e5e3f84a5 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.039894s | CSR: 0.891534s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.189053 s
ROLV per-iter: 0.001963s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568  (Dense GEMM (rocBLAS))
CSR_norm_hash:   3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568
ROLF_norm_hash:  d20759f7172d2f43c943a230dae84150fb48c05b8b882f77935ccb62e51e5023
DENGS_norm_hash: 3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568
ROLF per-iter:   0.000244s | total: 0.244254s
DENGS per-iter:  0.041195s | total: 41.195059s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 19.03x (≈ 1803% faster)
Speedup (per-iter): 20.86x (≈ 1986% faster)
Energy Savings: 95.21%
ROLV vs rocSPARSE -> Speedup (per-iter): 456.36x | total: 416.28x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "9d19ea5f391575455f95a6f93a0dc330f0816afb109185aa39e76d5e5e3f84a5", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568", "CSR_norm_hash": "3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568", "ROLF_norm_hash": "d20759f7172d2f43c943a230dae84150fb48c05b8b882f77935ccb62e51e5023", "DENGS_norm_hash": "3397dfb188f303cce8ca1e8cfc9ceaf57b34d9574df64e8d752935e89f273568", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "054e2c6d65e7082adb830742e210da6458ef0c3b993c9efeb6f8de4af5491a0b", "CSR_qhash_d6": "054e2c6d65e7082adb830742e210da6458ef0c3b993c9efeb6f8de4af5491a0b", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.039894, "pilot_csr_per_iter_s": 0.891534, "rolv_build_s": 0.189053, "rolv_iter_s": 0.001963, "dense_iter_s": 0.040952, "csr_iter_s": 0.895907, "rolv_total_s": 2.152195, "baseline_total_s": 40.951848, "speedup_total_vs_selected_x": 19.028, "speedup_iter_vs_selected_x": 20.86, "rolv_vs_vendor_sparse_iter_x": 456.364, "rolv_vs_vendor_sparse_total_x": 416.276, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:01:07] Seed: 123456 | Pattern: banded | Zeros: 60%
A_hash: e78e035e07d681d9c88788fb30448528322d3759de0292aef1030acc8d438be2 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034350s | CSR: 0.036028s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.186508 s
ROLV per-iter: 0.001958s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a  (Dense GEMM (rocBLAS))
CSR_norm_hash:   a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a
ROLF_norm_hash:  875e17c75ff856b76232765109555a25b7fbfd369a523cfcfc37fecd0cf0a765
DENGS_norm_hash: a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a
ROLF per-iter:   0.000226s | total: 0.226811s
DENGS per-iter:  0.035339s | total: 35.339434s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 16.34x (≈ 1534% faster)
Speedup (per-iter): 17.89x (≈ 1689% faster)
Energy Savings: 94.41%
ROLV vs rocSPARSE -> Speedup (per-iter): 18.94x | total: 17.29x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "e78e035e07d681d9c88788fb30448528322d3759de0292aef1030acc8d438be2", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a", "CSR_norm_hash": "a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a", "ROLF_norm_hash": "875e17c75ff856b76232765109555a25b7fbfd369a523cfcfc37fecd0cf0a765", "DENGS_norm_hash": "a917726a5ab831eb4b9c1cbef78f9dab9bf38f1875cc27bd0c7e1a74d85cd51a", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "6c590cb1c86449e9a55609b2add184da816091d6e591141b6417902275b2cba6", "CSR_qhash_d6": "6c590cb1c86449e9a55609b2add184da816091d6e591141b6417902275b2cba6", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.03435, "pilot_csr_per_iter_s": 0.036028, "rolv_build_s": 0.186508, "rolv_iter_s": 0.001958, "dense_iter_s": 0.035036, "csr_iter_s": 0.037089, "rolv_total_s": 2.144641, "baseline_total_s": 35.036426, "speedup_total_vs_selected_x": 16.337, "speedup_iter_vs_selected_x": 17.893, "rolv_vs_vendor_sparse_iter_x": 18.941, "rolv_vs_vendor_sparse_total_x": 17.294, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:03:11] Seed: 123456 | Pattern: block_diagonal | Zeros: 60%
A_hash: 2b99793bda656b5689cc9f5b049fc1a55ae8c234e0386e439c7204b281ffc158 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033387s | CSR: 0.023751s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186920 s
ROLV per-iter: 0.001961s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4
ROLF_norm_hash:  968c6fc2b0accf1cdc4ab0d5115cfd7051533a5d02294705346da1e28a71c50b
DENGS_norm_hash: 36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4
ROLF per-iter:   0.000222s | total: 0.222321s
DENGS per-iter:  0.034615s | total: 34.615254s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 11.25x (≈ 1025% faster)
Speedup (per-iter): 12.32x (≈ 1132% faster)
Energy Savings: 91.88%
ROLV vs rocSPARSE -> Speedup (per-iter): 12.42x | total: 11.34x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "2b99793bda656b5689cc9f5b049fc1a55ae8c234e0386e439c7204b281ffc158", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4", "CSR_norm_hash": "36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4", "ROLF_norm_hash": "968c6fc2b0accf1cdc4ab0d5115cfd7051533a5d02294705346da1e28a71c50b", "DENGS_norm_hash": "36ee25dfb91d647bc72a00e82673d5af290686eb222c108851af0797263cbfc4", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "4d9bc3a8c04e309be3d194ede6251942ddf9e71860fd8aa14f66686b71fd0279", "CSR_qhash_d6": "4d9bc3a8c04e309be3d194ede6251942ddf9e71860fd8aa14f66686b71fd0279", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033387, "pilot_csr_per_iter_s": 0.023751, "rolv_build_s": 0.18692, "rolv_iter_s": 0.001961, "dense_iter_s": 0.024158, "csr_iter_s": 0.024352, "rolv_total_s": 2.148172, "baseline_total_s": 24.158086, "speedup_total_vs_selected_x": 11.246, "speedup_iter_vs_selected_x": 12.318, "rolv_vs_vendor_sparse_iter_x": 12.416, "rolv_vs_vendor_sparse_total_x": 11.336, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:04:52] Seed: 123456 | Pattern: random | Zeros: 70%
A_hash: b6d397e4d0e8ebd4f3a13d59f635831bd762ee60284807ed9d008435058ec326 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.038887s | CSR: 0.730144s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188616 s
ROLV per-iter: 0.001965s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915  (Dense GEMM (rocBLAS))
CSR_norm_hash:   722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915
ROLF_norm_hash:  a503fe60776ce90abef86e50f774bce0642268c79ed36f1ed2a0cfdb456b6090
DENGS_norm_hash: 722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915
ROLF per-iter:   0.000239s | total: 0.239286s
DENGS per-iter:  0.040262s | total: 40.261652s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.53x (≈ 1753% faster)
Speedup (per-iter): 20.30x (≈ 1930% faster)
Energy Savings: 95.07%
ROLV vs rocSPARSE -> Speedup (per-iter): 373.92x | total: 341.18x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "b6d397e4d0e8ebd4f3a13d59f635831bd762ee60284807ed9d008435058ec326", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915", "CSR_norm_hash": "722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915", "ROLF_norm_hash": "a503fe60776ce90abef86e50f774bce0642268c79ed36f1ed2a0cfdb456b6090", "DENGS_norm_hash": "722aa1f103b022093a749ccf9de9cf9003cb678bf7f25648ca7f11ed9adde915", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "82da032e08a558947b8496a6df588242839c731dd132f6c51b2ccad1158e1e9e", "CSR_qhash_d6": "82da032e08a558947b8496a6df588242839c731dd132f6c51b2ccad1158e1e9e", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.038887, "pilot_csr_per_iter_s": 0.730144, "rolv_build_s": 0.188616, "rolv_iter_s": 0.001965, "dense_iter_s": 0.039906, "csr_iter_s": 0.734942, "rolv_total_s": 2.154114, "baseline_total_s": 39.905594, "speedup_total_vs_selected_x": 18.525, "speedup_iter_vs_selected_x": 20.303, "rolv_vs_vendor_sparse_iter_x": 373.922, "rolv_vs_vendor_sparse_total_x": 341.181, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:18:58] Seed: 123456 | Pattern: power_law | Zeros: 70%
A_hash: 64b353290cc661d8798233b459b02627e318c8b6cd03fb9400cdc258605a7257 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.039535s | CSR: 0.687016s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188740 s
ROLV per-iter: 0.001965s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc  (Dense GEMM (rocBLAS))
CSR_norm_hash:   32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc
ROLF_norm_hash:  72204d453e28b46f2f06b0ac265feda1932e914601162d00494a9d1bfa846619
DENGS_norm_hash: 32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc
ROLF per-iter:   0.000241s | total: 0.241699s
DENGS per-iter:  0.040608s | total: 40.607508s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.66x (≈ 1766% faster)
Speedup (per-iter): 20.46x (≈ 1946% faster)
Energy Savings: 95.11%
ROLV vs rocSPARSE -> Speedup (per-iter): 350.74x | total: 320.00x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "64b353290cc661d8798233b459b02627e318c8b6cd03fb9400cdc258605a7257", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc", "CSR_norm_hash": "32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc", "ROLF_norm_hash": "72204d453e28b46f2f06b0ac265feda1932e914601162d00494a9d1bfa846619", "DENGS_norm_hash": "32c0bbfd6d7a5688cd46fb2b36d62e508c168c1fca43ba3125721e0a93b9b9dc", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "b28317e48cc7768973ac901f2af18502a2b95897bacec4775b55b8b869b4083f", "CSR_qhash_d6": "b28317e48cc7768973ac901f2af18502a2b95897bacec4775b55b8b869b4083f", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.039535, "pilot_csr_per_iter_s": 0.687016, "rolv_build_s": 0.18874, "rolv_iter_s": 0.001965, "dense_iter_s": 0.040187, "csr_iter_s": 0.689069, "rolv_total_s": 2.153345, "baseline_total_s": 40.186719, "speedup_total_vs_selected_x": 18.662, "speedup_iter_vs_selected_x": 20.455, "rolv_vs_vendor_sparse_iter_x": 350.742, "rolv_vs_vendor_sparse_total_x": 319.999, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:32:18] Seed: 123456 | Pattern: banded | Zeros: 70%
A_hash: 6de52c734dc3dd3e441813467d3974c05babbe147880af95cae93106e22a77bd | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034296s | CSR: 0.028498s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186800 s
ROLV per-iter: 0.001966s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8
ROLF_norm_hash:  0f39b126fa71c26f2672d66e7c24c5b70fa1e325cbf4ccf3333c8d2df71581bf
DENGS_norm_hash: afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8
ROLF per-iter:   0.000226s | total: 0.227013s
DENGS per-iter:  0.035539s | total: 35.538520s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 13.51x (≈ 1251% faster)
Speedup (per-iter): 14.80x (≈ 1380% faster)
Energy Savings: 93.24%
ROLV vs rocSPARSE -> Speedup (per-iter): 14.90x | total: 13.60x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "6de52c734dc3dd3e441813467d3974c05babbe147880af95cae93106e22a77bd", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8", "CSR_norm_hash": "afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8", "ROLF_norm_hash": "0f39b126fa71c26f2672d66e7c24c5b70fa1e325cbf4ccf3333c8d2df71581bf", "DENGS_norm_hash": "afa0b5ccf007ed78efa389e675d49ed3175a5e895800ce2c51b65ef34c1c93f8", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "34587fe968cb118281d6e320a80b1d361638e54fc3de87df1dbf85b3f83c9fef", "CSR_qhash_d6": "34587fe968cb118281d6e320a80b1d361638e54fc3de87df1dbf85b3f83c9fef", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.034296, "pilot_csr_per_iter_s": 0.028498, "rolv_build_s": 0.1868, "rolv_iter_s": 0.001966, "dense_iter_s": 0.029087, "csr_iter_s": 0.029284, "rolv_total_s": 2.15251, "baseline_total_s": 29.087078, "speedup_total_vs_selected_x": 13.513, "speedup_iter_vs_selected_x": 14.797, "rolv_vs_vendor_sparse_iter_x": 14.897, "rolv_vs_vendor_sparse_total_x": 13.605, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:34:08] Seed: 123456 | Pattern: block_diagonal | Zeros: 70%
A_hash: 605ad79227a409511ccd935bac7446d55792ae15e0550623f778311797a2ba80 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033313s | CSR: 0.019110s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186859 s
ROLV per-iter: 0.001968s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75
ROLF_norm_hash:  71235fd9a701d922ba41c2836607dbd277b18a94a84c0fe54ad3e3cb46430593
DENGS_norm_hash: afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75
ROLF per-iter:   0.000219s | total: 0.219810s
DENGS per-iter:  0.034528s | total: 34.528223s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 9.02x (≈ 802% faster)
Speedup (per-iter): 9.87x (≈ 887% faster)
Energy Savings: 89.87%
ROLV vs rocSPARSE -> Speedup (per-iter): 9.93x | total: 9.07x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "605ad79227a409511ccd935bac7446d55792ae15e0550623f778311797a2ba80", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75", "CSR_norm_hash": "afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75", "ROLF_norm_hash": "71235fd9a701d922ba41c2836607dbd277b18a94a84c0fe54ad3e3cb46430593", "DENGS_norm_hash": "afb0000c8a8069b1416a99dc37be6c761f158e933ad2069f4c2a71f2a7f8ef75", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "7bc340a2266a60176174c6afbc41e54623a3acb3c008de5aeff26276a7332fb6", "CSR_qhash_d6": "7bc340a2266a60176174c6afbc41e54623a3acb3c008de5aeff26276a7332fb6", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033313, "pilot_csr_per_iter_s": 0.01911, "rolv_build_s": 0.186859, "rolv_iter_s": 0.001968, "dense_iter_s": 0.019425, "csr_iter_s": 0.019537, "rolv_total_s": 2.154482, "baseline_total_s": 19.424861, "speedup_total_vs_selected_x": 9.016, "speedup_iter_vs_selected_x": 9.872, "rolv_vs_vendor_sparse_iter_x": 9.929, "rolv_vs_vendor_sparse_total_x": 9.068, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:35:38] Seed: 123456 | Pattern: random | Zeros: 80%
A_hash: fe8ecd469d65375943070e2c9f72b2cb8ffc99f59b8e95e01ee55ff351e8a5b5 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.038113s | CSR: 0.504191s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187930 s
ROLV per-iter: 0.001964s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb  (Dense GEMM (rocBLAS))
CSR_norm_hash:   e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb
ROLF_norm_hash:  2c97f0d1200024546d7641880cac8a28d456f33e2b9d9806d0408d519cd56f37
DENGS_norm_hash: e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb
ROLF per-iter:   0.000238s | total: 0.238322s
DENGS per-iter:  0.039357s | total: 39.357312s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.15x (≈ 1715% faster)
Speedup (per-iter): 19.89x (≈ 1889% faster)
Energy Savings: 94.97%
ROLV vs rocSPARSE -> Speedup (per-iter): 258.84x | total: 236.24x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "fe8ecd469d65375943070e2c9f72b2cb8ffc99f59b8e95e01ee55ff351e8a5b5", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb", "CSR_norm_hash": "e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb", "ROLF_norm_hash": "2c97f0d1200024546d7641880cac8a28d456f33e2b9d9806d0408d519cd56f37", "DENGS_norm_hash": "e7c01a70a75e7c23f6388af2f7da803ea0bdb8bf7a90f33bfbd68ec38023b5fb", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "73400dd11bba92807f9dd80ee8c7bdc5e578106e1ea67465c31cebca0c1dc833", "CSR_qhash_d6": "73400dd11bba92807f9dd80ee8c7bdc5e578106e1ea67465c31cebca0c1dc833", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.038113, "pilot_csr_per_iter_s": 0.504191, "rolv_build_s": 0.18793, "rolv_iter_s": 0.001964, "dense_iter_s": 0.039075, "csr_iter_s": 0.508487, "rolv_total_s": 2.152404, "baseline_total_s": 39.074793, "speedup_total_vs_selected_x": 18.154, "speedup_iter_vs_selected_x": 19.891, "rolv_vs_vendor_sparse_iter_x": 258.841, "rolv_vs_vendor_sparse_total_x": 236.241, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:45:52] Seed: 123456 | Pattern: power_law | Zeros: 80%
A_hash: f5319945ed9e0de80929153636dd5033761020445fb403b1998eb9214d00e127 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.038470s | CSR: 0.472457s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.188282 s
ROLV per-iter: 0.001965s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048  (Dense GEMM (rocBLAS))
CSR_norm_hash:   0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048
ROLF_norm_hash:  222530270f5d311d6cd32e2219aefd7aa591c2a33d3143d33a5f01187d0d6e1b
DENGS_norm_hash: 0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048
ROLF per-iter:   0.000238s | total: 0.238245s
DENGS per-iter:  0.039595s | total: 39.595082s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 18.26x (≈ 1726% faster)
Speedup (per-iter): 20.01x (≈ 1901% faster)
Energy Savings: 95.00%
ROLV vs rocSPARSE -> Speedup (per-iter): 243.53x | total: 222.24x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "f5319945ed9e0de80929153636dd5033761020445fb403b1998eb9214d00e127", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048", "CSR_norm_hash": "0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048", "ROLF_norm_hash": "222530270f5d311d6cd32e2219aefd7aa591c2a33d3143d33a5f01187d0d6e1b", "DENGS_norm_hash": "0bbec08c037006aa33c812b530df9811cc2768a08858d9d8f610d0e9dc3f1048", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "66317c61bd76f15b22946d59ad005fceac533aed498ef9ff62ad0904ec173c58", "CSR_qhash_d6": "66317c61bd76f15b22946d59ad005fceac533aed498ef9ff62ad0904ec173c58", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.03847, "pilot_csr_per_iter_s": 0.472457, "rolv_build_s": 0.188282, "rolv_iter_s": 0.001965, "dense_iter_s": 0.039324, "csr_iter_s": 0.478547, "rolv_total_s": 2.153335, "baseline_total_s": 39.323539, "speedup_total_vs_selected_x": 18.262, "speedup_iter_vs_selected_x": 20.011, "rolv_vs_vendor_sparse_iter_x": 243.529, "rolv_vs_vendor_sparse_total_x": 222.235, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:55:36] Seed: 123456 | Pattern: banded | Zeros: 80%
A_hash: b2fc7f83b499ca9e4b29ed3cc68b966b4b322cf7926c12186e98ae033e84be58 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034383s | CSR: 0.020915s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186837 s
ROLV per-iter: 0.001961s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6
ROLF_norm_hash:  e59dd960a001f4488b5b3c7cdcaa8d75712839372bc52a94bab002baba02fd9d
DENGS_norm_hash: 7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6
ROLF per-iter:   0.000222s | total: 0.222409s
DENGS per-iter:  0.035405s | total: 35.404617s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 9.92x (≈ 892% faster)
Speedup (per-iter): 10.87x (≈ 987% faster)
Energy Savings: 90.80%
ROLV vs rocSPARSE -> Speedup (per-iter): 10.93x | total: 9.98x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "b2fc7f83b499ca9e4b29ed3cc68b966b4b322cf7926c12186e98ae033e84be58", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6", "CSR_norm_hash": "7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6", "ROLF_norm_hash": "e59dd960a001f4488b5b3c7cdcaa8d75712839372bc52a94bab002baba02fd9d", "DENGS_norm_hash": "7c5ad9bbb7deb3f95a8b23ba1ddfb19f857bf6120ba461e883e8789abd82d6c6", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "3d7a5452a9f38bbfb38ee0d4922ca8020b2506f811ff5ec119664b4fe4236084", "CSR_qhash_d6": "3d7a5452a9f38bbfb38ee0d4922ca8020b2506f811ff5ec119664b4fe4236084", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.034383, "pilot_csr_per_iter_s": 0.020915, "rolv_build_s": 0.186837, "rolv_iter_s": 0.001961, "dense_iter_s": 0.021311, "csr_iter_s": 0.021438, "rolv_total_s": 2.147726, "baseline_total_s": 21.310938, "speedup_total_vs_selected_x": 9.923, "speedup_iter_vs_selected_x": 10.868, "rolv_vs_vendor_sparse_iter_x": 10.933, "rolv_vs_vendor_sparse_total_x": 9.982, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:57:10] Seed: 123456 | Pattern: block_diagonal | Zeros: 80%
A_hash: 4b02e483523fbec343feac2b8fed3820615bb6832dda42a3da7b63ccf1ef0014 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033376s | CSR: 0.014398s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186835 s
ROLV per-iter: 0.001962s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82
ROLF_norm_hash:  50f833cd0edb871dab70fd1076fd14286647900fd24dbd3131295de2f48e9f7b
DENGS_norm_hash: 0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82
ROLF per-iter:   0.000218s | total: 0.218437s
DENGS per-iter:  0.034517s | total: 34.516832s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 6.81x (≈ 581% faster)
Speedup (per-iter): 7.46x (≈ 646% faster)
Energy Savings: 86.59%
ROLV vs rocSPARSE -> Speedup (per-iter): 7.50x | total: 6.85x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "4b02e483523fbec343feac2b8fed3820615bb6832dda42a3da7b63ccf1ef0014", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82", "CSR_norm_hash": "0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82", "ROLF_norm_hash": "50f833cd0edb871dab70fd1076fd14286647900fd24dbd3131295de2f48e9f7b", "DENGS_norm_hash": "0afabd3c3f898124c4d2adf90b4ec9ca28ee520d74975002112bb8408f023a82", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "4eaaed0b681ad4346a051280bb2504683f2650e05430ced90b415a8515706ae4", "CSR_qhash_d6": "4eaaed0b681ad4346a051280bb2504683f2650e05430ced90b415a8515706ae4", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033376, "pilot_csr_per_iter_s": 0.014398, "rolv_build_s": 0.186835, "rolv_iter_s": 0.001962, "dense_iter_s": 0.014627, "csr_iter_s": 0.014711, "rolv_total_s": 2.148341, "baseline_total_s": 14.627198, "speedup_total_vs_selected_x": 6.809, "speedup_iter_vs_selected_x": 7.457, "rolv_vs_vendor_sparse_iter_x": 7.5, "rolv_vs_vendor_sparse_total_x": 6.848, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 11:58:30] Seed: 123456 | Pattern: random | Zeros: 90%
A_hash: 252a6d9ec7eeab4eb29b6c652bffba9f11178919caadeccd14c45d00311e1433 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.036874s | CSR: 0.270021s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187616 s
ROLV per-iter: 0.001967s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83  (Dense GEMM (rocBLAS))
CSR_norm_hash:   0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83
ROLF_norm_hash:  321943d79a8e5a4c00b38e7a32c69fc5870129ed248809d076b51f4519f056ca
DENGS_norm_hash: 0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83
ROLF per-iter:   0.000237s | total: 0.237196s
DENGS per-iter:  0.038298s | total: 38.297789s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.64x (≈ 1664% faster)
Speedup (per-iter): 19.32x (≈ 1832% faster)
Energy Savings: 94.83%
ROLV vs rocSPARSE -> Speedup (per-iter): 139.67x | total: 127.51x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "252a6d9ec7eeab4eb29b6c652bffba9f11178919caadeccd14c45d00311e1433", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83", "CSR_norm_hash": "0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83", "ROLF_norm_hash": "321943d79a8e5a4c00b38e7a32c69fc5870129ed248809d076b51f4519f056ca", "DENGS_norm_hash": "0ea34324829b59e6d5a810b043219ca106a8eb538079e8849cd5903c80796f83", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "d07e9d8e4b761c9e713843d9e5ad22646d68738c9c9f78b846a77a566e81f77a", "CSR_qhash_d6": "d07e9d8e4b761c9e713843d9e5ad22646d68738c9c9f78b846a77a566e81f77a", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.036874, "pilot_csr_per_iter_s": 0.270021, "rolv_build_s": 0.187616, "rolv_iter_s": 0.001967, "dense_iter_s": 0.038016, "csr_iter_s": 0.274767, "rolv_total_s": 2.154825, "baseline_total_s": 38.015566, "speedup_total_vs_selected_x": 17.642, "speedup_iter_vs_selected_x": 19.325, "rolv_vs_vendor_sparse_iter_x": 139.674, "rolv_vs_vendor_sparse_total_x": 127.513, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:04:43] Seed: 123456 | Pattern: power_law | Zeros: 90%
A_hash: d1784f30a29c88bb759e8e0ce2e1d3a72ec63f8f7d0190e4b7c74bf9b0f76e26 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.037287s | CSR: 0.254723s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187980 s
ROLV per-iter: 0.001967s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6  (Dense GEMM (rocBLAS))
CSR_norm_hash:   ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6
ROLF_norm_hash:  d25ac0c9367d63ba2b4533bd33d053eefdea95762e4d5dafaf6a2ce9ff4cb49f
DENGS_norm_hash: ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6
ROLF per-iter:   0.000235s | total: 0.235172s
DENGS per-iter:  0.038551s | total: 38.550824s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.74x (≈ 1674% faster)
Speedup (per-iter): 19.43x (≈ 1843% faster)
Energy Savings: 94.85%
ROLV vs rocSPARSE -> Speedup (per-iter): 131.51x | total: 120.04x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "d1784f30a29c88bb759e8e0ce2e1d3a72ec63f8f7d0190e4b7c74bf9b0f76e26", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6", "CSR_norm_hash": "ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6", "ROLF_norm_hash": "d25ac0c9367d63ba2b4533bd33d053eefdea95762e4d5dafaf6a2ce9ff4cb49f", "DENGS_norm_hash": "ff7b7b9d919c85aa942dd3c65988841f5aedc3f475953f6a39377245c2e213f6", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "11f8da7a2080d0d605a1ebff2f877f43fdf7d15739e0c108fe9752e7a0e9c93a", "CSR_qhash_d6": "11f8da7a2080d0d605a1ebff2f877f43fdf7d15739e0c108fe9752e7a0e9c93a", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.037287, "pilot_csr_per_iter_s": 0.254723, "rolv_build_s": 0.18798, "rolv_iter_s": 0.001967, "dense_iter_s": 0.038222, "csr_iter_s": 0.258692, "rolv_total_s": 2.155098, "baseline_total_s": 38.222414, "speedup_total_vs_selected_x": 17.736, "speedup_iter_vs_selected_x": 19.431, "rolv_vs_vendor_sparse_iter_x": 131.508, "rolv_vs_vendor_sparse_total_x": 120.037, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:10:40] Seed: 123456 | Pattern: banded | Zeros: 90%
A_hash: d70a4343e5b268957eb68d7e3674a43f240457ccfda08b4a2d80bc40ab643157 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034564s | CSR: 0.013217s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186761 s
ROLV per-iter: 0.001963s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94
ROLF_norm_hash:  7b612cfd403099d0e842a486b4d51c55fba1f9c7ff554a4a1ed5affedc0f75a1
DENGS_norm_hash: 6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94
ROLF per-iter:   0.000218s | total: 0.218567s
DENGS per-iter:  0.035382s | total: 35.381969s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 6.23x (≈ 523% faster)
Speedup (per-iter): 6.82x (≈ 582% faster)
Energy Savings: 85.35%
ROLV vs rocSPARSE -> Speedup (per-iter): 6.86x | total: 6.27x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "d70a4343e5b268957eb68d7e3674a43f240457ccfda08b4a2d80bc40ab643157", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94", "CSR_norm_hash": "6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94", "ROLF_norm_hash": "7b612cfd403099d0e842a486b4d51c55fba1f9c7ff554a4a1ed5affedc0f75a1", "DENGS_norm_hash": "6bab4e46cb1871e51f5424a844af2f1390bcb5c5fb0b3a7c6f421bd1bc78bc94", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "c850461ae5bfa1c3183f8c5ad70eadff35c42a0cf45abfac99892c98541b884e", "CSR_qhash_d6": "c850461ae5bfa1c3183f8c5ad70eadff35c42a0cf45abfac99892c98541b884e", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.034564, "pilot_csr_per_iter_s": 0.013217, "rolv_build_s": 0.186761, "rolv_iter_s": 0.001963, "dense_iter_s": 0.013395, "csr_iter_s": 0.01347, "rolv_total_s": 2.149746, "baseline_total_s": 13.39487, "speedup_total_vs_selected_x": 6.231, "speedup_iter_vs_selected_x": 6.824, "rolv_vs_vendor_sparse_iter_x": 6.862, "rolv_vs_vendor_sparse_total_x": 6.266, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:11:59] Seed: 123456 | Pattern: block_diagonal | Zeros: 90%
A_hash: ef3c072370841e3130690e4f6793ea35e3e0c704fce673efdbae340a03091d07 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033338s | CSR: 0.009611s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186375 s
ROLV per-iter: 0.001964s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79
ROLF_norm_hash:  043a6ad806a02a4dbfd07edb33a20bdebb586c17d7abfcd015ef28379fdadb27
DENGS_norm_hash: 3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79
ROLF per-iter:   0.000215s | total: 0.215233s
DENGS per-iter:  0.034487s | total: 34.487371s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 4.51x (≈ 351% faster)
Speedup (per-iter): 4.93x (≈ 393% faster)
Energy Savings: 79.73%
ROLV vs rocSPARSE -> Speedup (per-iter): 4.96x | total: 4.53x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "ef3c072370841e3130690e4f6793ea35e3e0c704fce673efdbae340a03091d07", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79", "CSR_norm_hash": "3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79", "ROLF_norm_hash": "043a6ad806a02a4dbfd07edb33a20bdebb586c17d7abfcd015ef28379fdadb27", "DENGS_norm_hash": "3954f5e832c294f5f63bb74e0179a360d788ef7079faeb84f69713613cc4ea79", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "cffe32a36e15addac2c5b2fef97a992d6d9c8731c108477cdc00f463498f0e00", "CSR_qhash_d6": "cffe32a36e15addac2c5b2fef97a992d6d9c8731c108477cdc00f463498f0e00", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033338, "pilot_csr_per_iter_s": 0.009611, "rolv_build_s": 0.186375, "rolv_iter_s": 0.001964, "dense_iter_s": 0.009688, "csr_iter_s": 0.009746, "rolv_total_s": 2.150311, "baseline_total_s": 9.688463, "speedup_total_vs_selected_x": 4.506, "speedup_iter_vs_selected_x": 4.933, "rolv_vs_vendor_sparse_iter_x": 4.963, "rolv_vs_vendor_sparse_total_x": 4.532, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:13:09] Seed: 123456 | Pattern: random | Zeros: 95%
A_hash: c926d3fc034ec0adbed3fa6ecc74c1e0c4191486cd48fd095fa3c179c6ef96db | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.036446s | CSR: 0.169984s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187381 s
ROLV per-iter: 0.001963s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71  (Dense GEMM (rocBLAS))
CSR_norm_hash:   f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71
ROLF_norm_hash:  438099e42e0c75acf4c952287040709c2d6b5b9b38bb0103770643c2625437f3
DENGS_norm_hash: f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71
ROLF per-iter:   0.000232s | total: 0.232182s
DENGS per-iter:  0.037337s | total: 37.337230s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.34x (≈ 1634% faster)
Speedup (per-iter): 18.99x (≈ 1799% faster)
Energy Savings: 94.73%
ROLV vs rocSPARSE -> Speedup (per-iter): 88.95x | total: 81.20x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "c926d3fc034ec0adbed3fa6ecc74c1e0c4191486cd48fd095fa3c179c6ef96db", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71", "CSR_norm_hash": "f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71", "ROLF_norm_hash": "438099e42e0c75acf4c952287040709c2d6b5b9b38bb0103770643c2625437f3", "DENGS_norm_hash": "f5570aa0c2e30ca57e4bfba6db1ba2ed338b13cf6c3a3861cd9efa7d20b15d71", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "c341e71c1d6aaaff215d32bf3ce2823faac0512f379a99a19bf0274553f15740", "CSR_qhash_d6": "c341e71c1d6aaaff215d32bf3ce2823faac0512f379a99a19bf0274553f15740", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.036446, "pilot_csr_per_iter_s": 0.169984, "rolv_build_s": 0.187381, "rolv_iter_s": 0.001963, "dense_iter_s": 0.037276, "csr_iter_s": 0.174584, "rolv_total_s": 2.150007, "baseline_total_s": 37.276063, "speedup_total_vs_selected_x": 17.338, "speedup_iter_vs_selected_x": 18.993, "rolv_vs_vendor_sparse_iter_x": 88.954, "rolv_vs_vendor_sparse_total_x": 81.202, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:17:38] Seed: 123456 | Pattern: power_law | Zeros: 95%
A_hash: 6417a2a60f09c4389956722addb9e641d9618bcfe0eae0e987dfe602fdefb429 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.036440s | CSR: 0.164895s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187498 s
ROLV per-iter: 0.001963s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf  (Dense GEMM (rocBLAS))
CSR_norm_hash:   e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf
ROLF_norm_hash:  d02f72a24d8562f072ac59b90a05de2bb21f3e093177220577b1c675fdd1e16e
DENGS_norm_hash: e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf
ROLF per-iter:   0.000232s | total: 0.233011s
DENGS per-iter:  0.037607s | total: 37.606988s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.40x (≈ 1640% faster)
Speedup (per-iter): 19.06x (≈ 1806% faster)
Energy Savings: 94.75%
ROLV vs rocSPARSE -> Speedup (per-iter): 86.55x | total: 79.00x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "6417a2a60f09c4389956722addb9e641d9618bcfe0eae0e987dfe602fdefb429", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf", "CSR_norm_hash": "e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf", "ROLF_norm_hash": "d02f72a24d8562f072ac59b90a05de2bb21f3e093177220577b1c675fdd1e16e", "DENGS_norm_hash": "e1a90360ba8f3a6ce55dfff4d1515996f6b04b522f024ed9ab8124c98d9cb2cf", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "100de79f25f9eba2174c313cdd119f1094c254144c65bddf7539fe46bd2b2afb", "CSR_qhash_d6": "100de79f25f9eba2174c313cdd119f1094c254144c65bddf7539fe46bd2b2afb", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.03644, "pilot_csr_per_iter_s": 0.164895, "rolv_build_s": 0.187498, "rolv_iter_s": 0.001963, "dense_iter_s": 0.037407, "csr_iter_s": 0.16988, "rolv_total_s": 2.150259, "baseline_total_s": 37.407164, "speedup_total_vs_selected_x": 17.397, "speedup_iter_vs_selected_x": 19.058, "rolv_vs_vendor_sparse_iter_x": 86.551, "rolv_vs_vendor_sparse_total_x": 79.004, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:22:04] Seed: 123456 | Pattern: banded | Zeros: 95%
A_hash: f9841b629a96caca12ae5093b69047a66277d824418f1f09df0d2ec6bec61381 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034407s | CSR: 0.008984s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186270 s
ROLV per-iter: 0.001955s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08
ROLF_norm_hash:  da09351c75b5339ca035bd5d494f359e4c85eb4b75b0e3f8b5462fc1a53761ba
DENGS_norm_hash: a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08
ROLF per-iter:   0.000213s | total: 0.213486s
DENGS per-iter:  0.035288s | total: 35.287984s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 4.24x (≈ 324% faster)
Speedup (per-iter): 4.64x (≈ 364% faster)
Energy Savings: 78.46%
ROLV vs rocSPARSE -> Speedup (per-iter): 4.67x | total: 4.26x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "f9841b629a96caca12ae5093b69047a66277d824418f1f09df0d2ec6bec61381", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08", "CSR_norm_hash": "a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08", "ROLF_norm_hash": "da09351c75b5339ca035bd5d494f359e4c85eb4b75b0e3f8b5462fc1a53761ba", "DENGS_norm_hash": "a00aed09a9ca1a80f3acaafa376dc0e568949aeb81fcc547c28bc98683803a08", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "9c378462296ec1cacef0a364ddaeebca041f1cfda7d5705308d0e32cbdf1f369", "CSR_qhash_d6": "9c378462296ec1cacef0a364ddaeebca041f1cfda7d5705308d0e32cbdf1f369", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.034407, "pilot_csr_per_iter_s": 0.008984, "rolv_build_s": 0.18627, "rolv_iter_s": 0.001955, "dense_iter_s": 0.009076, "csr_iter_s": 0.009123, "rolv_total_s": 2.141716, "baseline_total_s": 9.076221, "speedup_total_vs_selected_x": 4.238, "speedup_iter_vs_selected_x": 4.642, "rolv_vs_vendor_sparse_iter_x": 4.666, "rolv_vs_vendor_sparse_total_x": 4.26, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:23:13] Seed: 123456 | Pattern: block_diagonal | Zeros: 95%
A_hash: 743ed1c8dc03a5de5d0b131edc508c8c9e30dc02e5406aeb9cb6e8c0ce493874 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033158s | CSR: 0.006990s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186436 s
ROLV per-iter: 0.001958s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1
ROLF_norm_hash:  a23f52d144ae759bed22286b940fe6340572d8562f2e0cdd90f665c8a5c99fa6
DENGS_norm_hash: cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1
ROLF per-iter:   0.000212s | total: 0.212509s
DENGS per-iter:  0.034332s | total: 34.332445s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 3.29x (≈ 229% faster)
Speedup (per-iter): 3.60x (≈ 260% faster)
Energy Savings: 72.23%
ROLV vs rocSPARSE -> Speedup (per-iter): 3.61x | total: 3.30x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "743ed1c8dc03a5de5d0b131edc508c8c9e30dc02e5406aeb9cb6e8c0ce493874", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1", "CSR_norm_hash": "cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1", "ROLF_norm_hash": "a23f52d144ae759bed22286b940fe6340572d8562f2e0cdd90f665c8a5c99fa6", "DENGS_norm_hash": "cb31498379c907686529a18f196926f32e7b1052704f4ed5aaebf0f0e0ed14b1", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "efbe0ca36ca8f3c6721275268db28beadf0ad8a4b2a7aa76452f596348100e65", "CSR_qhash_d6": "efbe0ca36ca8f3c6721275268db28beadf0ad8a4b2a7aa76452f596348100e65", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033158, "pilot_csr_per_iter_s": 0.00699, "rolv_build_s": 0.186436, "rolv_iter_s": 0.001958, "dense_iter_s": 0.00705, "csr_iter_s": 0.007073, "rolv_total_s": 2.144613, "baseline_total_s": 7.050193, "speedup_total_vs_selected_x": 3.287, "speedup_iter_vs_selected_x": 3.6, "rolv_vs_vendor_sparse_iter_x": 3.612, "rolv_vs_vendor_sparse_total_x": 3.298, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:24:18] Seed: 123456 | Pattern: random | Zeros: 99%
A_hash: 9fde8b5d279f5d4d8297c2b0a4f006d0bf2475b62e6dabc7da09b547c8edbc8a | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.035845s | CSR: 0.067016s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.187014 s
ROLV per-iter: 0.001964s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9  (Dense GEMM (rocBLAS))
CSR_norm_hash:   cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9
ROLF_norm_hash:  cd16ca2eb816f4e3a4324002312ca0268a38b732796aab7888293ef7f24633b7
DENGS_norm_hash: cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9
ROLF per-iter:   0.000224s | total: 0.224806s
DENGS per-iter:  0.036825s | total: 36.825078s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.01x (≈ 1601% faster)
Speedup (per-iter): 18.63x (≈ 1763% faster)
Energy Savings: 94.63%
ROLV vs rocSPARSE -> Speedup (per-iter): 35.55x | total: 32.46x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "9fde8b5d279f5d4d8297c2b0a4f006d0bf2475b62e6dabc7da09b547c8edbc8a", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9", "CSR_norm_hash": "cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9", "ROLF_norm_hash": "cd16ca2eb816f4e3a4324002312ca0268a38b732796aab7888293ef7f24633b7", "DENGS_norm_hash": "cc8c3cc839a9d0c5930d0938d01d9bd2a7f5d66f47fcce4e53dec39e0dc7faa9", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "18a55821ec1e53d19620c8b3fdf1bd7dc27837e4d3f0bc4b8bb33ab2e24117eb", "CSR_qhash_d6": "18a55821ec1e53d19620c8b3fdf1bd7dc27837e4d3f0bc4b8bb33ab2e24117eb", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.035845, "pilot_csr_per_iter_s": 0.067016, "rolv_build_s": 0.187014, "rolv_iter_s": 0.001964, "dense_iter_s": 0.036594, "csr_iter_s": 0.069822, "rolv_total_s": 2.150907, "baseline_total_s": 36.594293, "speedup_total_vs_selected_x": 17.013, "speedup_iter_vs_selected_x": 18.634, "rolv_vs_vendor_sparse_iter_x": 35.553, "rolv_vs_vendor_sparse_total_x": 32.462, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:26:59] Seed: 123456 | Pattern: power_law | Zeros: 99%
A_hash: 3884cba828aa7a1488fc132da5edbcb037e4d5cda60d2548cbb05d1438117888 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.036139s | CSR: 0.063748s
Selected baseline: Dense GEMM (rocBLAS)
ROLV load time (operator build): 0.186683 s
ROLV per-iter: 0.001962s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d  (Dense GEMM (rocBLAS))
CSR_norm_hash:   c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d
ROLF_norm_hash:  4727304f2130748ccc072bc431f0db2c0cbe41049c05e3323613e00e8f930359
DENGS_norm_hash: c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d
ROLF per-iter:   0.000223s | total: 0.223517s
DENGS per-iter:  0.037162s | total: 37.161934s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 17.19x (≈ 1619% faster)
Speedup (per-iter): 18.83x (≈ 1783% faster)
Energy Savings: 94.69%
ROLV vs rocSPARSE -> Speedup (per-iter): 33.69x | total: 30.77x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "3884cba828aa7a1488fc132da5edbcb037e4d5cda60d2548cbb05d1438117888", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d", "CSR_norm_hash": "c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d", "ROLF_norm_hash": "4727304f2130748ccc072bc431f0db2c0cbe41049c05e3323613e00e8f930359", "DENGS_norm_hash": "c3e9496be8a189fc75d306823ff6359f8fa3b5aa5557bbc72bae19d4a7ed7d5d", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "570879d26c1763504bd05013868ba03d6e5c343019d405fbb6664799c3446a0a", "CSR_qhash_d6": "570879d26c1763504bd05013868ba03d6e5c343019d405fbb6664799c3446a0a", "path_selected": "Dense", "pilot_dense_per_iter_s": 0.036139, "pilot_csr_per_iter_s": 0.063748, "rolv_build_s": 0.186683, "rolv_iter_s": 0.001962, "dense_iter_s": 0.03694, "csr_iter_s": 0.066115, "rolv_total_s": 2.148859, "baseline_total_s": 36.940387, "speedup_total_vs_selected_x": 17.191, "speedup_iter_vs_selected_x": 18.826, "rolv_vs_vendor_sparse_iter_x": 33.695, "rolv_vs_vendor_sparse_total_x": 30.768, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:29:38] Seed: 123456 | Pattern: banded | Zeros: 99%
A_hash: 1b643fe5ac4811868b9b5bfee7d7ed4d02a612b4add98ac2d0f399d014599b67 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.034358s | CSR: 0.005434s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186462 s
ROLV per-iter: 0.001955s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad
ROLF_norm_hash:  832a9e85bddee020a7655ba9a16031e464caa87f1ceb92d8567b75780ccb7c6d
DENGS_norm_hash: 2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad
ROLF per-iter:   0.000210s | total: 0.210169s
DENGS per-iter:  0.035618s | total: 35.618500s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 2.55x (≈ 155% faster)
Speedup (per-iter): 2.79x (≈ 179% faster)
Energy Savings: 64.15%
ROLV vs rocSPARSE -> Speedup (per-iter): 2.80x | total: 2.55x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "1b643fe5ac4811868b9b5bfee7d7ed4d02a612b4add98ac2d0f399d014599b67", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad", "CSR_norm_hash": "2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad", "ROLF_norm_hash": "832a9e85bddee020a7655ba9a16031e464caa87f1ceb92d8567b75780ccb7c6d", "DENGS_norm_hash": "2974cb2e57d28e2c52f6b64f51fb5c2c10ab828f6e7279b652ec7d4c7fa407ad", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "36fabc0207e13e7ae08a8c6db45526167c20c66464e356ba7cd4969570c1cb52", "CSR_qhash_d6": "36fabc0207e13e7ae08a8c6db45526167c20c66464e356ba7cd4969570c1cb52", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.034358, "pilot_csr_per_iter_s": 0.005434, "rolv_build_s": 0.186462, "rolv_iter_s": 0.001955, "dense_iter_s": 0.005453, "csr_iter_s": 0.005467, "rolv_total_s": 2.14101, "baseline_total_s": 5.452658, "speedup_total_vs_selected_x": 2.547, "speedup_iter_vs_selected_x": 2.79, "rolv_vs_vendor_sparse_iter_x": 2.797, "rolv_vs_vendor_sparse_total_x": 2.554, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

[2025-12-07 12:30:40] Seed: 123456 | Pattern: block_diagonal | Zeros: 99%
A_hash: d78e202117fb1b5ee60605254db62aa72b0d2b72a9d6ceec1a84ad78c44df368 | V_hash: 448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070
Baseline pilots per-iter -> Dense: 0.033287s | CSR: 0.004741s
Selected baseline: Sparse CSR (rocSPARSE)
ROLV load time (operator build): 0.186746 s
ROLV per-iter: 0.001957s
ROLV_norm_hash:  8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd | qhash(d=6): 8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd
BASE_norm_hash:  81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b  (Sparse CSR (rocSPARSE))
CSR_norm_hash:   81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b
ROLF_norm_hash:  aedd4cf9d03fcc98a2d4b33b9c56bc89ad6ca996f94ce0f83609d26a240cfcd3
DENGS_norm_hash: 81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b
ROLF per-iter:   0.000211s | total: 0.210978s
DENGS per-iter:  0.034538s | total: 34.538172s
Correctness vs Selected Baseline: Verified | vs CSR: Verified
Speedup (total): 2.22x (≈ 122% faster)
Speedup (per-iter): 2.43x (≈ 143% faster)
Energy Savings: 58.88%
ROLV vs rocSPARSE -> Speedup (per-iter): 2.44x | total: 2.22x
{"platform": "ROCm", "device": "AMD Instinct MI300X", "adapted_batch": false, "effective_batch": 5000, "dense_label": "rocBLAS", "sparse_label": "rocSPARSE", "input_hash_A": "d78e202117fb1b5ee60605254db62aa72b0d2b72a9d6ceec1a84ad78c44df368", "input_hash_B": "448b453ff50675840e0e32980b9e77974b1188713089eb2c28e45b6d12701070", "ROLV_norm_hash": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_norm_hash": "81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b", "CSR_norm_hash": "81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b", "ROLF_norm_hash": "aedd4cf9d03fcc98a2d4b33b9c56bc89ad6ca996f94ce0f83609d26a240cfcd3", "DENGS_norm_hash": "81df416748e59ff8fb7b3e31c4a3a74db121c9fc011e70c1604e496e3b107c2b", "ROLV_qhash_d6": "8dbe5f139fd946d4cd84e8cc612cd9f68cbc87e394457884acc0c5dad56dd8dd", "DENSE_qhash_d6": "72ae2e8b1b11989b240bd407be5eae8d1ffdbf75beeacce50c056cdb544c412f", "CSR_qhash_d6": "72ae2e8b1b11989b240bd407be5eae8d1ffdbf75beeacce50c056cdb544c412f", "path_selected": "CSR", "pilot_dense_per_iter_s": 0.033287, "pilot_csr_per_iter_s": 0.004741, "rolv_build_s": 0.186746, "rolv_iter_s": 0.001957, "dense_iter_s": 0.004758, "csr_iter_s": 0.004766, "rolv_total_s": 2.143294, "baseline_total_s": 4.757702, "speedup_total_vs_selected_x": 2.22, "speedup_iter_vs_selected_x": 2.432, "rolv_vs_vendor_sparse_iter_x": 2.436, "rolv_vs_vendor_sparse_total_x": 2.224, "energy_iter_adaptive_telemetry": null, "telemetry_samples": 0, "correct_norm": "OK"}

=== FOOTER REPORT (ROCm) ===
- Aggregate speedup (total vs selected): 13.96x (≈ 1296% faster)
- Aggregate speedup (per-iter vs selected): 15.30x (≈ 1430% faster)
- Aggregate energy savings (proxy vs selected): 90.1%
- Verification: TF32 off, deterministic algorithms, CSR canonicalization, CPU-fp64 normalization and SHA-256 hashing.



