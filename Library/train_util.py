import torch
import torch.nn as nn

def save_checkpoint(model, optimizer, scheduler,  iteration, filepath):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer=None, scheduler=None, filepath="",map_location='cuda:0'):
    if filepath=="":
        return 0
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    iteration = checkpoint['iteration']
    print(f"successfully load model from {filepath}")
    return iteration
    
def process_pose_sample(model, sample, device_id):
    """
    모델과 샘플을 받아서 예측 포즈와 그라운드 트루스 포즈 간의 손실을 계산하는 함수
    
    Args:
        model (nn.Module): VONet 모델
        sample (dict): 배치 샘플로, 'img1', 'img2', 'intrinsic', 'motion' 키를 포함
        device (torch.device): 연산에 사용할 디바이스(CPU 또는 GPU)
    
    Returns:
        tuple: (total_loss, trans_loss, rot_loss)
    """
    img1 = sample['img1'].to(device_id)         # [B, 3, H, W]
    img2 = sample['img2'].to(device_id)         # [B, 3, H, W]
    intrinsic = sample['intrinsic'].to(device_id)  # [B, 3, 3]
    motion_gt = sample['motion'].to(device_id)  # [B, 6] -> [tx, ty, tz, rx, ry, rz]
    
    model_output = model([img1, img2, intrinsic])
    flow_pred, pose_pred = model_output      # flow_pred: [B, 2, H, W], pose_pred: [B, 6]
    
    total_loss, trans_loss, rot_loss = calculate_pose_loss(pose_pred, motion_gt)
    
    return total_loss, trans_loss, rot_loss

def process_whole_sample(model,sample,lambda_flow,device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']
    intrinsic_layer = sample['intrinsic']
        
    # forward------------------------------------------------------------------
    flow, relative_motion = model([img1,img2,intrinsic_layer])


    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']
    motions_gt = sample['motion']
    flow_loss = model.module.flowNet.get_loss(flow,flow_gt,small_scale=True)
    pose_loss,trans_loss,rot_loss = model.module.flowPoseNet.linear_norm_trans_loss(relative_motion, motions_gt)
    total_loss = flow_loss*lambda_flow + pose_loss
    
    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss

def process_flow_sample(model, sample, device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']
        
    # forward------------------------------------------------------------------
    flow = model([img1,img2])
    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']
    flow_loss =  model.module.get_loss(flow,flow_gt,small_scale=True)
    return flow_loss

def process_flowpose_sample(model,sample,device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    intrinsic_layer = sample['intrinsic']
    flow_gt = sample['flow']
        
    flow_input = torch.cat( ( flow_gt, intrinsic_layer ), dim=1 ) 
    # forward------------------------------------------------------------------
    relative_motion = model(flow_input)


    # loss calculation---------------------------------------------------------
    motions_gt = sample['motion']
    total_loss,trans_loss,rot_loss = calculate_pose_loss(relative_motion, motions_gt,device_id)
    
    return total_loss,trans_loss,rot_loss

'''def calculate_pose_loss( relative_motion, motions_gt,device_id='cuda:0'):
    
    # Translation loss with normalization
    epsilon = 1e-6
    T_pred = relative_motion[:, :3]
    T_gt = motions_gt[:, :3]
    T_pred_norm = T_pred / torch.max(torch.norm(T_pred, dim=1, keepdim=True), torch.tensor(epsilon).to(device_id))
    T_gt_norm = T_gt / torch.max(torch.norm(T_gt, dim=1, keepdim=True), torch.tensor(epsilon).to(device_id))
    trans_loss = torch.nn.functional.mse_loss(T_pred_norm, T_gt_norm)
    
    # Simple Rotation loss
    R_pred = relative_motion[:, 3:]
    R_gt = motions_gt[:, 3:]
    rot_loss = torch.nn.functional.mse_loss(R_pred, R_gt)

    # Overall motion loss
    pose_loss = trans_loss + rot_loss


    return pose_loss,trans_loss,rot_loss'''

def calculate_pose_loss(pose_pred, pose_gt, trans_weight=1.0, rot_weight=1.0):
    """
    예측 포즈와 그라운드 트루스 포즈 간의 손실을 계산
    
    Args:
        pose_pred (torch.Tensor): 예측된 포즈 [B, 6] -> [tx, ty, tz, rx, ry, rz]
        pose_gt (torch.Tensor): 그라운드 트루스 포즈 [B, 6] -> [tx, ty, tz, rx, ry, rz]
        trans_weight (float): 변환 손실 가중치
        rot_weight (float): 회전 손실 가중치
    
    Returns:
        tuple: (total_loss, trans_loss, rot_loss)
    """
    # 변환 및 회전 부분 분리
    trans_pred = pose_pred[:, :3]
    rot_pred = pose_pred[:, 3:]
    
    trans_gt = pose_gt[:, :3]
    rot_gt = pose_gt[:, 3:]
    
    # 변환 손실 계산 (MSE Loss)
    trans_loss_fn = nn.MSELoss()
    trans_loss = trans_loss_fn(trans_pred, trans_gt)
    
    # 회전 손실 계산 (MSE Loss)
    # 회전 부분은 각도(rad)로 표현된다고 가정
    rot_loss_fn = nn.MSELoss()
    rot_loss = rot_loss_fn(rot_pred, rot_gt)
    
    # 총 손실 계산
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss
    
    return total_loss, trans_loss, rot_loss


def test_pose_batch(model, sample):
    model.eval()
    # inputs-------------------------------------------------------------------
    flow_gt = sample['flow']
    motions_gt = sample['motion']
    intrinsic_gt = sample['intrinsic']
    # forward------------------------------------------------------------------
    with torch.no_grad():
        # batch shapes are [batch_size,channels, Height,Width]
        # So concant on dimension 1 not 0
        flow_input = torch.cat((flow_gt, intrinsic_gt), dim=1)
        relative_motion = model(flow_input)
        total_loss,trans_loss,rot_loss = calculate_pose_loss(relative_motion, motions_gt)
       
    relative_motion = relative_motion.cpu().numpy()
    # if 'motion' in sample:
    #     motions_gt = sample['motion'].cpu().numpy()
    #     scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #     trans_est = relative_motion[:,:3]
    #     trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
    #     relative_motion[:,:3] = trans_est 
    # else:
    #     print('    scale is not given, using 1 as the default scale value..')

    return relative_motion,total_loss,trans_loss,rot_loss