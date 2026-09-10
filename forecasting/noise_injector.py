# noise_injector.py
import torch

def apply_gaussian_noise(x: torch.Tensor, snr_db: float):
    """
    Inference 시 입력 텐서에 지정된 SNR(dB) 강도의 가우시안 노이즈를 주입합니다.
    
    Args:
        x (torch.Tensor): 원본 입력 데이터, shape [Batch, Sequence_Length, Channel]
        snr_db (float): 신호 대 잡음비 (dB 단위). 값이 작을수록 노이즈가 강해집니다.
                        None이거나 100 이상이면 노이즈를 추가하지 않고 원본을 반환합니다.
                        
    Returns:
        torch.Tensor: 노이즈가 추가된 입력 데이터
    """
    if snr_db is None or snr_db >= 100:
        return x
    
    # 1. 시퀀스 길이(dim=1)를 기준으로 각 배치/채널별 신호의 분산(Power) 계산
    # unbiased=False를 사용하여 정확한 모집단 분산을 구함
    signal_power = torch.var(x, dim=1, keepdim=True, unbiased=False)
    
    # 신호가 완전히 상수(constant)일 경우 0으로 나누는 것을 방지
    signal_power = torch.clamp(signal_power, min=1e-8)
    
    # 2. SNR 공식을 역산하여 필요한 노이즈의 분산(Power) 계산
    # SNR(dB) = 10 * log10(Signal_Power / Noise_Power)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    
    # 3. 평균 0, 분산 1인 가우시안 노이즈 생성 후 계산된 noise_power에 맞게 스케일링
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    
    # 4. 원본 신호에 노이즈 합성
    return x + noise