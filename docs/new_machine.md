# note for training in new machine
- change torch nvidia to newest version
```
    nvcr.io/nvidia/pytorch:21.08-py3
```
- checkout `update-docker` branch
```
    git checkout update-docker
```
- minor change inside docker container :lib `/opt/conda/lib/python3.8/site-packages/torch/_tensor.py`
    - edit above lib
    ```
        vim /opt/conda/lib/python3.8/site-packages/torch/_tensor.py
    ```
    - at line 647 change from `return self.numpy()` to `return self.cpu().numpy()`