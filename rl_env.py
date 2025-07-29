import gym, numpy as np
from gym import spaces
class OnePosEnv(gym.Env):
    def __init__(self, feat_mat: np.ndarray, price_arr: np.ndarray, tau_set=(3,6,9,12)):
        super().__init__()
        self.feat_mat, self.price, self.tau_set = feat_mat, price_arr, tau_set
        self.max_step=len(price_arr)-1
        n_feat=feat_mat.shape[1]
        self.observation_space=spaces.Box(-np.inf, np.inf, shape=(n_feat+4,), dtype=np.float32)
        self.action_space=spaces.MultiDiscrete([4,len(tau_set)])
        self.reset()
    def reset(self):
        self.step_idx=0; self.pos_flag=0; self.entry_price=0.; self.tau_target=0; self.time_in_pos=0
        return self._obs()
    def _obs(self):
        extra=np.array([self.pos_flag,self.entry_price,self.time_in_pos,self.tau_target],dtype=np.float32)
        return np.concatenate([self.feat_mat[self.step_idx],extra])
    def step(self,action):
        act,tau_idx=action
        reward,done=0.,False
        price=self.price[self.step_idx]
        if self.pos_flag==0 and act==3: act=0
        if self.pos_flag!=0 and act in (1,2): act=0
        if self.pos_flag==0 and act in (1,2):
            self.pos_flag=1 if act==1 else -1
            self.entry_price=price
            self.tau_target=self.tau_set[tau_idx]
            self.time_in_pos=0
        elif self.pos_flag!=0 and act==3:
            pnl=(price-self.entry_price)*self.pos_flag
            reward=pnl-0.001
            self.pos_flag=0; self.entry_price=0.; self.tau_target=0; self.time_in_pos=0
        self.step_idx+=1
        if self.step_idx>=self.max_step: done=True
        if self.pos_flag!=0:
            self.time_in_pos+=1
            if self.time_in_pos>=self.tau_target:
                pnl=(price-self.entry_price)*self.pos_flag
                reward=pnl-0.001
                self.pos_flag=0; self.entry_price=0.; self.tau_target=0; self.time_in_pos=0
        reward-=0.0001*(self.time_in_pos>0)
        return self._obs(), reward, done, {}