import os, json

for algo in ['sac', 'ddqn']:
    for s in ['42', '123', '2024']:
        d = f'checkpoints_{algo}_seed{s}_0716'
        p = os.path.join(d, 'training_curves.json')
        if not os.path.exists(p):
            print(f'{d}: MISSING')
            continue
        c = json.load(open(p, encoding='utf-8'))
        ep = c['episode']
        rw = c['ep_reward']
        best_pth = os.path.join(d, f'{algo}_best.pth')
        ckpt_pth = os.path.join(d, f'{algo}_checkpoint.pth')
        best_exists = "YES" if os.path.exists(best_pth) else "NO"
        ckpt_exists = "YES" if os.path.exists(ckpt_pth) else "NO"
        # eval_eff_peak final value
        ev = c.get('eval_eff_peak', [])
        ev_final = ev[-1] if ev else -1
        print(f'{d:45s} ep=[{ep[0]:>4},{ep[-1]:>4}]  final_rw={rw[-1]:7.2f}  '
              f'eval_ep={ev_final:.3f}  best={best_exists}  ckpt={ckpt_exists}')
