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
    #trans_loss_fn = nn.MSELoss()
    #trans_loss = trans_loss_fn(trans_pred_norm, trans_gt_norm)
    trans_loss = F.l1_loss(trans_pred_norm, trans_gt_norm)

    # rotation loss
    #rot_loss_fn = nn.MSELoss()
    #rot_loss = rot_loss_fn(rot_est, rot_gt)
    rot_loss = F.l1_loss(rot_est, rot_gt)

    # pose loss
    pose_loss = trans_loss + rot_loss

    return pose_loss, trans_loss, rot_loss

def whole_loss_function(model, sample, lambda_flow, epsilon, device='cuda:0'):
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

def flow_loss_function(model, sample, device='cuda:0'):
    sample = {k: v.to(device) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']

    # forward------------------------------------------------------------------
    flow_est = model([img1,img2])

    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']

    flow_loss = flow_loss_fn(flow_est, flow_gt)
    
    return flow_loss

def pose_loss_function(model, sample, epsilon, device='cuda:0'):
    sample = {k: v.to(device) for k, v in sample.items()}
    # inputs-------------------------------------------------------------------
    intrinsic_layer = sample['intrinsic']
    flow_gt = sample['flow']
        
    flow_input = torch.cat( ( flow_gt, intrinsic_layer ), dim=1 )

    # forward------------------------------------------------------------------
    motion_est = model(flow_input)


    # loss calculation---------------------------------------------------------
    motion_gt = sample['motion']

    pose_loss, trans_loss, rot_loss = pose_loss_fn(motion_est, motion_gt, epsilon)
    
    return pose_loss, trans_loss, rot_loss

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