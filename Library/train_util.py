import torch
import torch.nn as nn
import torch.nn.functional as F
    
def flow_loss_fn(flow_est, flow_gt):
    
    #flow_loss_fn = F.l1_loss()
    flow_loss = F.l1_loss(flow_est, flow_gt)

    return flow_loss

def pose_loss_fn(motion_est, motion_gt, epsilon=1e-6):
    # input motion -> transformation & rotation
    trans_est = motion_est[:, :3]
    rot_est = motion_est[:, 3:]
    
    trans_gt = motion_gt[:, :3]
    rot_gt = motion_gt[:, 3:]

    # transformation loss
    trans_pred_norm = trans_est / torch.max(trans_est.norm(dim=1, keepdim=True), torch.tensor(epsilon).to(trans_est.device))
    trans_gt_norm = trans_gt / torch.max(trans_gt.norm(dim=1, keepdim=True), torch.tensor(epsilon).to(trans_gt.device))
    #trans_loss_fn = F.l1_loss()
    trans_loss = F.l1_loss(trans_pred_norm, trans_gt_norm)

    # rotation loss
    #rot_loss_fn = F.l1_loss()
    rot_loss = F.l1_loss(rot_est, rot_gt)

    # pose loss
    pose_loss = trans_loss + rot_loss

    return pose_loss, trans_loss, rot_loss

def loss_function(model, sample, lambda_flow, epsilon, device='cuda:0'):
    sample = {k: v.to(device) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']
    intrinsic_layer = sample['intrinsic']
        
    # forward------------------------------------------------------------------
    flow_est, motion_est = model([img1,img2,intrinsic_layer])

    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']
    motion_gt = sample['motion']

    # pose loss
    pose_loss, trans_loss, rot_loss = pose_loss_fn(motion_est, motion_gt, epsilon)

    # flow loss
    flow_loss = flow_loss_fn(flow_est, flow_gt)

    # total loss
    total_loss = flow_loss*lambda_flow + pose_loss
    
    return total_loss, flow_loss, pose_loss, trans_loss, rot_loss

def process_pose_sample(model, sample, device_id):
    """
    모델과 샘플을 받아서 예측 포즈와 그라운드 트루스 포즈 간의 손실을 계산하는 함수
    
    Args:
        model (nn.Module): VONet 모델
        sample (dict): 배치 샘플로, 'img1', 'img2', 'intrinsic', 'motion' 키를 포함
        device (torch.device): 연산에 사용할 디바이스(CPU 또는 GPU)
    
    Returns:
        tuple: loss
    """
    img1 = sample['img1'].to(device_id)         # [B, 3, H, W]
    img2 = sample['img2'].to(device_id)         # [B, 3, H, W]
    intrinsic = sample['intrinsic'].to(device_id)  # [B, 3, 3]
    motion_gt = sample['motion'].to(device_id)  # [B, 6] -> [tx, ty, tz, rx, ry, rz]
    
    model_output = model([img1, img2, intrinsic])
    flow_pred, pose_pred = model_output      # flow_pred: [B, 2, H, W], pose_pred: [B, 6]
    
    # Only Total Pose Loss
    # loss_fn = nn.MSELoss()
    # loss = loss_fn(pose_pred, motion_gt)

    # Using Total Pose loss: trans Loss + Rotate Loss
    total_loss, trans_loss, rot_loss = calculate_pose_loss(pose_pred, motion_gt)

    
    return total_loss, trans_loss, rot_loss
    #return loss

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
    trans_pred = pose_pred[:, :3]
    rot_pred = pose_pred[:, 3:]
    
    trans_gt = pose_gt[:, :3]
    rot_gt = pose_gt[:, 3:]
    
    # trans loss
    trans_loss_fn = nn.MSELoss()
    trans_loss = trans_loss_fn(trans_pred, trans_gt)
    
    # rotation loss
    rot_loss_fn = nn.MSELoss()
    rot_loss = rot_loss_fn(rot_pred, rot_gt)
    
    total_loss = trans_weight*trans_loss + rot_weight*rot_loss
    
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
    get_loss = nn.MSELoss()
    flow_loss = get_loss(flow,flow_gt)
    pose_loss,trans_loss,rot_loss = calculate_pose_loss(relative_motion, motions_gt)
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