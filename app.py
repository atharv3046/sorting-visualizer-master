import streamlit as st
import time
import random
import pandas as pd
import altair as alt


def bubble_sort(arr):
    n = len(arr)
    steps = []
    working_indices = []
    for i in range(n):
        for j in range(0, n - i - 1):
            working_indices.append([j, j+1])
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            steps.append(arr.copy())
    return steps, working_indices

def insertion_sort(arr):
    steps = []
    working_indices = []
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        working_indices.append([i, j])
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            steps.append(arr.copy())
            working_indices.append([j, j+1])
            j -= 1
        arr[j + 1] = key
        steps.append(arr.copy())
        working_indices.append([j+1])
    return steps, working_indices

def selection_sort(arr):
    steps = []
    working_indices = []
    for i in range(len(arr)):
        min_idx = i
        working_indices.append([i, min_idx])
        for j in range(i + 1, len(arr)):
            working_indices.append([j, min_idx])
            if arr[j] < arr[min_idx]:
                min_idx = j
                working_indices[-1] = [j, min_idx]
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        steps.append(arr.copy())
        working_indices.append([i, min_idx])
    return steps, working_indices

def merge_sort(arr):
    steps = []
    working_indices = []
    
    def merge(arr, l, m, r):
        left = arr[l:m+1]
        right = arr[m+1:r+1]
        i = j = 0
        k = l
        
        while i < len(left) and j < len(right):
            working_indices.append([k, l+i, m+1+j])
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            steps.append(arr.copy())
            k += 1
            
        while i < len(left):
            working_indices.append([k, l+i])
            arr[k] = left[i]
            i += 1
            k += 1
            steps.append(arr.copy())
            
        while j < len(right):
            working_indices.append([k, m+1+j])
            arr[k] = right[j]
            j += 1
            k += 1
            steps.append(arr.copy())

    def merge_sort_rec(arr, l, r):
        if l < r:
            m = (l + r) // 2
            merge_sort_rec(arr, l, m)
            merge_sort_rec(arr, m + 1, r)
            merge(arr, l, m, r)
    
    merge_sort_rec(arr, 0, len(arr) - 1)
    return steps, working_indices

def quick_sort(arr):
    steps = []
    working_indices = []
    
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        working_indices.append([high])  
        
        for j in range(low, high):
            working_indices.append([j, high, i+1])  
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
            steps.append(arr.copy())
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append(arr.copy())
        working_indices.append([i+1, high])
        return i + 1

    def quick_sort_rec(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_rec(low, pi - 1)
            quick_sort_rec(pi + 1, high)
    
    quick_sort_rec(0, len(arr) - 1)
    return steps, working_indices

def heap_sort(arr):
    steps = []
    working_indices = []
    
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        
        working_indices.append([i, l, r])
        
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
            
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            steps.append(arr.copy())
            working_indices.append([i, largest])
            heapify(arr, n, largest)
    
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
        
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        steps.append(arr.copy())
        working_indices.append([i, 0])
        heapify(arr, i, 0)
        
    return steps, working_indices



def create_colored_barchart(data, highlight_indices=None):
    """Create a bar chart with only the currently compared indices highlighted in red"""
    if highlight_indices is None:
        highlight_indices = []

    flat_highlight_indices = set()
    for item in highlight_indices:
        if isinstance(item, list):
            flat_highlight_indices.update(item)
        else:
            flat_highlight_indices.add(item)

    df = pd.DataFrame({
        'index': range(len(data)),
        'value': data,
        'color': ['red' if i in flat_highlight_indices else 'steelblue' for i in range(len(data))]
    })

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('index:O', title='Array Index', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('color:N', scale=None, legend=None)
        )
        .properties(width=600, height=400)
    )

    return chart




st.title("🎨 Sorting Algorithm Visualizer")
st.write("This project visualizes how different sorting algorithms operate, made completely in Python using Streamlit.")

algo = st.selectbox(
    "Choose an Algorithm",
    ["Bubble Sort", "Insertion Sort", "Selection Sort", "Merge Sort", "Quick Sort", "Heap Sort"]
)

size = st.slider("Select array size", min_value=5, max_value=20, value=10)
speed = st.slider("Animation speed", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

if 'original_arr' not in st.session_state:
    st.session_state.original_arr = random.sample(range(1, 100), size)

arr = st.session_state.original_arr.copy()

if st.button("Generate New Array"):
    st.session_state.original_arr = random.sample(range(1, 100), size)
    st.rerun()

st.write("*Original Array:*", arr)

if st.button("Start Visualization"):
    st.write(f"*Visualizing {algo}*")
    
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
    
    if algo == "Bubble Sort":
        steps, working_indices = bubble_sort(arr.copy())
    elif algo == "Insertion Sort":
        steps, working_indices = insertion_sort(arr.copy())
    elif algo == "Selection Sort":
        steps, working_indices = selection_sort(arr.copy())
    elif algo == "Merge Sort":
        steps, working_indices = merge_sort(arr.copy())
    elif algo == "Quick Sort":
        steps, working_indices = quick_sort(arr.copy())
    elif algo == "Heap Sort":
        steps, working_indices = heap_sort(arr.copy())

    for i, (step, indices) in enumerate(zip(steps, working_indices)):
        with info_placeholder.container():
            st.write(f"*Step {i+1}/{len(steps)}:* Working on indices {indices}")
        
        chart = create_colored_barchart(step, indices)
        chart_placeholder.altair_chart(chart, use_container_width=True)
        
        time.sleep(1.0 / speed)
    
    st.success(" Sorting Completed!")
    st.write("*Final Sorted Array:*", steps[-1] if steps else arr)


    # ---- Put near the top of your file (or keep your existing one) ----
ALGO_DETAILS = {
    "Bubble Sort": {
        "title": "🫧 Bubble Sort",
        "description": """

*How it works (Steps):*

1. Start from the first element in the list.
2. Compare the current element with the next element.
3. If the current element is greater, swap them.
4. Move to the next pair and repeat.
5. After one full pass, the largest element moves to the end.
6. Repeat for the remaining unsorted part.
7. Stop when a pass makes no swaps.

*Pseudocode:*

bubbleSort(arr)
    n = length(arr)
    for i = 0 to n - 1
        for j = 0 to n - i - 2
            if arr[j] > arr[j + 1]
                swap(arr[j], arr[j + 1])



*Time Complexity:* 
Best *O(n)*, 
Average *O(n²)*, 
Worst *O(n²)*  

*Space Complexity:* *O(1)*  
*Stable:* ✅ Yes
"""
    },
    "Insertion Sort": {
        "title": "🧩 Insertion Sort",
        "description": """
*How it works (Steps):*
1. Start from the second element (since the first is already “sorted”).
2. Store this element in a variable called key.
3. Compare key with the elements in the sorted portion (to its left).
4. Shift every element that is greater than key one position to the right.
5. Insert key at the correct position in the sorted portion.
6. Repeat steps 2–5 for every element until the whole array is sorted.

*Pseudocode:*


insertionSort(arr)
    n = length(arr)
    for i = 1 to n - 1
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key




*Time Complexity:* Best *O(n), Average **O(n²), Worst **O(n²)*  
*Space Complexity:* *O(1)*  
*Stable:* ✅ Yes
"""
    },
    "Selection Sort": {
        "title": "🎯 Selection Sort",
        "description": """
*How it works (Steps):*

1. Start from the first element (index 0).
2. Assume it’s the minimum element (min_index = i).
3. Go through the rest of the list to find the actual smallest element.
4. When found, swap it with the element at min_index.
5. Move to the next position and repeat steps 2–4 until the whole array is sorted.

*Pseudocode:*


selectionSort(arr)
    n = length(arr)
    for i = 0 to n - 1
        min_index = i
        for j = i + 1 to n - 1
            if arr[j] < arr[min_index]
                min_index = j
        swap(arr[i], arr[min_index])

 


*Time Complexity:* Best/Average/Worst *O(n²)*  
*Space Complexity:* *O(1)*  
*Stable:* ❌ No
"""
    },
    "Merge Sort": {
        "title": "🧵 Merge Sort",
        "description": """
*How it works (Steps):*

1. Divide the array into two halves.
2. Recursively sort both halves.
3. Merge the two sorted halves into a single sorted array.
4. Continue until all halves are merged.

*Pseudocode:*


mergeSort(arr, l, r):
if l < r:
    m = (l + r) // 2
    mergeSort(arr, l, m)
    mergeSort(arr, m+1, r)
    merge(arr, l, m, r)



*Time Complexity:* Best/Average/Worst *O(n log n)*  
*Space Complexity:* *O(n)*  
*Stable:* ✅ Yes
"""
    },
    "Quick Sort": {
        "title": "⚡ Quick Sort",
        "description": """

*How it works (Steps):*

1. Pick a pivot element.
2. Partition the array — elements smaller than pivot go left, larger go right.
3. Recursively apply the same logic to both halves.
4. Combine results — the array is sorted.

*Pseudocode:*


quickSort(arr, low, high):
if low < high:
    pi = partition(arr, low, high)
    quickSort(arr, low, pi-1)
    quickSort(arr, pi+1, high)



*Time Complexity:* Best/Average *O(n log n), Worst **O(n²)*  
*Space Complexity:* *O(log n)* (avg recursion)  
*Stable:* ❌ No
"""
    },
    "Heap Sort": {
        "title": "🏗 Heap Sort",
        "description": """
*How it works (Steps):*
1. Build a max heap from the array.
2. Swap the first (largest) element with the last element.
3. Reduce the heap size and heapify the root.
4. Repeat until the heap size is 1.

*Pseudocode:*

heapSort(arr):
buildMaxHeap(arr)
for i = n-1 down to 1:
    swap(arr[0], arr[i])
    heapify(arr, 0, i)



*Time Complexity:* Best/Average/Worst *O(n log n)*  
*Space Complexity:* *O(1)*  
*Stable:* ❌ No
"""
    },
}

# ❌ DELETE this whole block:
# st.markdown("""
# ###  Algorithm Explanations:
# - *Bubble Sort*: ...
# - *Insertion Sort*: ...
# - ...
# """)


    # --------------------------------------------------------
# 🔍 Compare All Algorithms
# --------------------------------------------------------
with st.expander("⚖ Compare All Algorithms", expanded=False):
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Insertion Sort": insertion_sort,
        "Selection Sort": selection_sort,
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Heap Sort": heap_sort
    }

    if st.button("Run Comparison"):
        results = []
        for name, algo_func in algorithms.items():
            test_arr = arr.copy()
            start_time = time.time()
            steps, indices = algo_func(test_arr)
            end_time = time.time()
            results.append({
                "Algorithm": name,
                "Time (s)": round(end_time - start_time, 4),
                "Steps": len(steps),
                "Efficiency": round((len(arr) * len(arr)) / len(steps), 2) if steps else 0
            })

        results_df = pd.DataFrame(results)

        st.markdown("### ⚖ Algorithm Performance Comparison")
        st.dataframe(
            results_df.style
            .highlight_min(subset=["Time (s)", "Steps"], color='lightgreen')
            .highlight_max(subset=["Efficiency"], color='lightblue')
        )

        final_df = pd.DataFrame({
            'index': range(len(steps[-1])),
            'value': steps[-1],
            'color': ['green'] * len(steps[-1])
        })

        final_chart = alt.Chart(final_df).mark_bar().encode(
            x=alt.X('index:O', title='Array Index'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('color:N', scale=None, legend=None)
        ).properties(
            width=600,
            height=400,
            title="Final Sorted Array"
        )

        st.altair_chart(final_chart, use_container_width=True)


# --- Detailed explanation for the selected algorithm (AFTER sorting completes) ---
st.markdown("---")
st.markdown(f"### {ALGO_DETAILS[algo]['title']}")
# Collapsible panel optional; set expanded=True/False as you like
with st.expander("See detailed steps, pseudocode, complexities, and stability", expanded=True):
    st.markdown(ALGO_DETAILS[algo]['description'])