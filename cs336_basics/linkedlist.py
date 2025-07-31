
class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
    def __repr__(self):
        return f"DoublyNode({self.data})"

class LinkedList:
    def __init__(self, items =None):
        
        self.head = None
        self.tail = None
        self.length = 0
        if items is not None:
            self.from_list(items)
        
    def is_empty(self):
        return self.head is None

    # initialize from a list 
    def from_list(self, a:list):
        for ele in a:
            self.append(ele)

        self.length = len(a)

    def append(self, data):
        new_node = DoublyNode(data)
        if self.is_empty():
            self.tail = self.head = new_node
            self.length += 1
        else: 
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
            self.length += 1

    
    def __len__(self):
        return self.length


    def __repr__(self):
        """链表的字符串表示"""
        cur = self.head
        nodes = []
        while cur is not None:
            nodes.append(str(cur))
            cur = cur.next
        return " -> ".join(nodes) if nodes else "Empty LinkedList"

if __name__ == '__main__':
    list = LinkedList()
    list.from_list([1,2,3,4,5])
    print(list)
    list.append(100)
    print(list)