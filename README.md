# Keras Image Retrieval

네이버 AI 해커톤 2018_Ai Vision
TEAM : GodGam

---

This repo is keras implemented model for Image-retrieval task and implementing [ArcFace](<https://arxiv.org/abs/1801.07698>) and [m-head ensemble](<https://arxiv.org/pdf/1804.00382.pdf>).

**ArcFace: Additive Angular Margin Loss for Deep Face Recognition**

$\mathcal{L} = - \dfrac{1}{N} \displaystyle\sum_{i=1}^{N}\log{\dfrac{\exp{s(\cos(m_1 \theta_{y_i} + m_2) -m_3 )}}   {\exp{s (\cos (m_1\theta_{y_i} + m_2 ) - m_3) + \sum_{j=1,j\neq y_i}^n \exp{s \cos \theta_j}  }}   }$

$m_1, m_2, m_3​$ are hyper-parameter for combined loss, which is including SphereFace, ArcFace and CosFace. Here I set 1, 0.5, 0 for each.
