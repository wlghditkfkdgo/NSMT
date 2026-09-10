"""
test.py 수정 방법
==================
test.py 의 `get_model_complexity_info(...)` 블록을 아래 코드로 교체합니다.
SNN 모델이면 기존대로 syops 를 측정하고,
iTransformer (비-SNN) 이면 MAC만 측정하고 AC/firing_rate 는 0으로 채웁니다.

교체 대상 범위 (test.py 기준):
────────────────────────────────
        model.train_mode = 'testing'
        functional.reset_net(model)
        ops, params, fr = get_model_complexity_info(...)
        file_out.close()
────────────────────────────────
위 블록 전체를 아래로 교체하세요.
"""

# ────────────────────────────────────────────────────────────────────────
# test.py 교체 스니펫 (들여쓰기는 기존 with torch.no_grad() 블록 안 수준)
# ────────────────────────────────────────────────────────────────────────

SNIPPET = '''
        # ── complexity / energy 측정 ─────────────────────────────────────
        IS_SNN = args.model != 'iTransformer'

        if IS_SNN:
            model.train_mode = 'testing'
            functional.reset_net(model)
            ops, params, fr = get_model_complexity_info(
                                        model=model,
                                        input_res=(input_res,),
                                        dataloader=loader,
                                        as_strings=False,
                                        print_per_layer_stat=True,
                                        verbose=False,
                                        ost=file_out,
                                    )
        else:
            # iTransformer : thop 으로 MAC 만 측정, AC / firing_rate = 0
            try:
                from thop import profile as thop_profile
                dummy = torch.zeros(1, args.seq_len, args.c_in).to(args.device)
                mac_ops, n_params = thop_profile(model, inputs=(dummy,), verbose=False)
                ops    = (mac_ops, 0, mac_ops)   # (total, AC, MAC)
                params = n_params
                fr     = 0.0
            except ImportError:
                # thop 도 없으면 파라미터 수만 수동 집계
                n_params = sum(p.numel() for p in model.parameters())
                ops    = (0, 0, 0)
                params = n_params
                fr     = 0.0
                print("[test] thop not installed — MAC ops set to 0.")

        file_out.close()
        # ─────────────────────────────────────────────────────────────────
'''

print(SNIPPET)
