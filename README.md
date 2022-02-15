## Competition 2 성능 평가 검증을 위한 실행 가이드

**바탕화면에 있는 pycharm IDE를 실행시켜 test.py 파일을 열어줍니다. 이때 프로젝트의 경로는 F:\Competition_No2\code 이며, test.py의 경로는 F:\Competition_No2\code\OCR\test.py 입니다.**

![image](https://user-images.githubusercontent.com/30248006/154059434-58e6d91c-ef74-4a01-8ed2-8bc76f3b45bb.png)

**test data set 01, 02, 03번의 이미지, 레이블 정보가 포함된 txt파일, test 결과를 확인할 폴더 각각의 경로를 다음과 같이 설정하여 실행합니다.**

**- test_data_path : test 이미지가 들어있는 폴더 경로**  
**- test_label_path : test 레이블 정보가 들어있는 txt파일 경로**  
**- test_result_path : test 결과 csv 파일이 저장되는 폴더 경로**  

```Python
# test_data_path = 'F:/Competition_No2/datasets/test/01'
# test_label_path = 'F:/Competition_No2/datasets/test/01/gt_test_01.txt'
# test_result_path = 'F:/Competition_No2/test_result/test/01'
#
# test_data_path = 'F:/Competition_No2/datasets/test/02'
# test_label_path = 'F:/Competition_No2/datasets/test/02/gt_test_02.txt'
# test_result_path = 'F:/Competition_No2/test_result/test/02'

test_data_path = 'F:/Competition_No2/datasets/test/03'
test_label_path = 'F:/Competition_No2/datasets/test/03/gt_test_03.txt'
test_result_path = 'F:/Competition_No2/test_result/test/03'
```
**모델은 세가지 데이터셋 01, 02, 03을 모두 예측할 수 있는 통합 모델이며, 각 데이터셋에 대해서 테스트를 한번씩 수행합니다. 이때 테스트를 수행하지 않는 데이터 셋에 대해서는 주석처리 해야하며, 총 3번 test.py 파일을 실행합니다. (test.py 파일 실행 단축키는 Ctrl + Enter로 설정되어 있으며, data set 01에 대한 테스트를 수행할 경우 02, 03번에 해당하는 코드 주석처리**

**실행을 완료했으면, 각 테스트 데이터 셋 01, 02, 03의 결과를 test_result_path에서 csv 파일을 통해 확인합니다.**

![image](https://user-images.githubusercontent.com/30248006/154065361-fac6821d-8822-4635-9954-e77c550ba849.png)

