#!/usr/bin/env python
# coding: utf-8

# In[4]:


def calculate(raw_data, predict):
    
    temp = raw_data.loc[predict.index]
    temp['pred'] = predict['pred']
    temp['y2'] = (~(temp['y1'].astype(bool))).astype(int)
    temp['pred2'] = (~(temp['pred'].astype(bool))).astype(int)

    temp['[기존]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['y2']
    temp['[기존]원금손실'] = temp['대출실행금액'] * temp['y1']
    
    temp['[예상]대출금액'] = temp['대출실행금액'] * temp['pred2']
    temp['[예상]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['pred2'] * temp['y2']
    temp['[예상]원금손실'] = temp['대출실행금액'] * temp['pred2'] * temp['y1']

    return temp

    # print('======= 예측 전 =======')
    # print('총 대출액 : ', format(temp['대출실행금액'].sum(),','))
    # print('이자 총액 : ', format(temp['이자수익'].sum(), ','))
    # print('원금 손실 : ', format(temp['원금손실'].sum(), ','))
    # print('=======================')
    # print("")
    # print('======= 예측 후 =======')
    # print('총 대출액 : ', format((temp['대출실행금액'] * temp['pred2']).sum(),','))
    # print('이자 총액 : ', format(temp['[예상]이자수익'].sum(), ','))
    # print('원금 손실 : ', format(temp['[예상]원금손실'].sum(), ','))
    # print('=======================')

    # print("")
    # print('========= 결과 =========')
    # print('대출액 증감 : ', format(temp['대출실행금액'].sum() - (temp['대출실행금액'] * temp['pred2']).sum(),','))
    # print('이자 증감 : ', format(temp['[예상]이자수익'].sum() - temp['이자수익'].sum(), ','))
    # print('원금 손실 증감 : ', format(temp['원금손실'].sum() - temp['[예상]원금손실'].sum(), ','))
    # print('결과 : ', format(temp['이자수익'].sum() - temp['[예상]이자수익'].sum() + temp['원금손실'].sum() - temp['[예상]원금손실'].sum(), ','))
    # print('=======================')


# In[ ]:


def calculate_4(raw_data, predict):
    
    temp = raw_data.loc[predict.index]
    temp['pred'] = predict['pred']
    temp['y2'] = (~(temp['y1'].astype(bool))).astype(int)
    temp['pred2'] = (~(temp['pred'].astype(bool))).astype(int)

    temp['[기존]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['y2']
    temp['[기존]원금손실'] = temp['대출실행금액'] * temp['y1']
    
    temp['[예상]대출금액'] = temp['대출실행금액'] * temp['pred2']
    temp['[예상]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['pred2'] * temp['y2']
    temp['[예상]원금손실'] = temp['대출실행금액'] * temp['pred2'] * temp['y1']
    
    temp['[동일기존]이자수익'] = temp['동일기존_대출실행금액'] * temp['금리'] * 0.01 * temp['y2']
    temp['[동일기존]원금손실'] = temp['동일기존_대출실행금액'] * temp['y1']
    
    temp['[예상]대출금액'] = temp['대출실행금액'] * temp['pred2']
    temp['[예상]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['pred2'] * temp['y2']
    temp['[예상]원금손실'] = temp['대출실행금액'] * temp['pred2'] * temp['y1']

    return temp


# In[2]:


if __name__ == "__main__":
    print("A")


# In[ ]:



