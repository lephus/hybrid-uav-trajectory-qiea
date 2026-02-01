# Hướng dẫn tạo Maps để QIEA thể hiện sự vượt trội

## Vấn đề hiện tại

Từ kết quả test, ta thấy QIEA chưa thể hiện được sự vượt trội so với các thuật toán cổ điển:
- AStar, ThetaStar, Dijkstra cho cùng path length
- QIEA variants cũng cho cùng path length (chưa cải thiện)

## Tại sao QIEA chưa thể hiện được?

QIEA hoạt động tốt nhất khi:
1. **Nhiều local optima**: Classical algorithms có thể bị stuck ở local optimum, QIEA có thể explore nhiều solutions
2. **Narrow passages**: QIEA có thể smooth path tốt hơn qua các narrow passages
3. **Multiple alternative paths**: QIEA có thể tìm được path ngắn nhất trong nhiều options
4. **Longer paths với nhiều waypoints**: QIEA có thể optimize tốt hơn với nhiều waypoints

## Giải pháp: Map Type m4 (QIEA Challenge)

Map type **m4** được thiết kế đặc biệt để QIEA thể hiện được sự vượt trội:

### Đặc điểm của m4:
- **Maze-like structure**: Tạo ra nhiều alternative paths với độ dài khác nhau
- **Narrow passages**: Yêu cầu path smoothing, QIEA có thể optimize tốt hơn
- **Multiple routes**: Classical algorithms có thể chọn route dài hơn, QIEA có thể tìm route ngắn hơn
- **Local optima**: Tạo ra nhiều local optima để QIEA explore tốt hơn

### Cách sử dụng:

```bash
# Tạo map m4 nhỏ (50x50) - nhanh để test
python3 map_generator_nn.py --type m4 --size 50 --uavs 3

# Tạo map m4 lớn (100x100) - tốt hơn cho QIEA
python3 map_generator_nn.py --type m4 --size 100 --uavs 5

# Test trên map m4
python3 test_nn_paths.py --map maps/nn_destinations/m4_100x100_5nn.json --viz --save-viz
```

## Các tham số ảnh hưởng đến hiệu quả QIEA

### 1. Map Size
- **Nhỏ (20x20, 50x50)**: Paths ngắn, QIEA ít có cơ hội optimize
- **Lớn (100x100, 150x150)**: Paths dài hơn, nhiều waypoints, QIEA có thể optimize tốt hơn
- **Khuyến nghị**: Sử dụng size >= 100 để QIEA có cơ hội thể hiện

### 2. Số lượng UAVs
- **Ít (3-5)**: Dễ test, nhưng ít data points
- **Nhiều (10+)**: Nhiều test cases, nhưng tốn thời gian
- **Khuyến nghị**: 5-10 UAVs là tốt nhất

### 3. Map Complexity
- **m1 (sparse)**: Quá đơn giản, QIEA không có lợi thế
- **m2 (dense)**: Nhiều obstacles nhưng paths vẫn đơn giản
- **m3 (trap)**: Khó nhưng chỉ có 1-2 paths, QIEA ít có lợi thế
- **m4 (QIEA challenge)**: Được thiết kế đặc biệt cho QIEA
- **Khuyến nghị**: Sử dụng m4 để test QIEA

## Tối ưu hóa QIEA parameters

Nếu QIEA vẫn chưa thể hiện tốt, có thể tăng parameters:

```python
# Trong test_nn_paths.py, có thể tạo planners với parameters cao hơn:
AStarQIEA(obstacles, width, height, 
          qiea_population_size=50,  # Tăng từ 30
          qiea_max_generations=100)  # Tăng từ 50
```

## Kỳ vọng kết quả trên m4

Trên map m4, bạn nên thấy:
- **QIEA variants có path length ngắn hơn** classical algorithms (5-15%)
- **QIEA có thể tìm được paths tốt hơn** qua narrow passages
- **Computation time cao hơn** nhưng đổi lại path tốt hơn

## Ví dụ workflow

```bash
# 1. Tạo map m4
python3 map_generator_nn.py --type m4 --size 100 --uavs 5

# 2. Test và visualize
python3 test_nn_paths.py --map maps/nn_destinations/m4_100x100_5nn.json --viz --save-viz

# 3. So sánh kết quả
# Xem trong maps/visualizations/ để so sánh paths
```

## Lưu ý

- Map m4 có nhiều obstacles hơn (maze-like structure), nên generation có thể mất thời gian hơn
- QIEA variants sẽ tốn thời gian hơn (do optimization), nhưng nên cho kết quả tốt hơn
- Nếu QIEA vẫn chưa thể hiện tốt, thử tăng map size hoặc số UAVs
