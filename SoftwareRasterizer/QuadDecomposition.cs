using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static VectorMath;

using Vertex = Int32;

public readonly struct Graph
{
    public int numVertices()
    {
        return m_adjacencyList.Length;
    }

    public readonly List<Vertex>[] m_adjacencyList;

    public Graph(int length)
    {
        m_adjacencyList = new List<Vertex>[length];

        for (int i = 0; i < m_adjacencyList.Length; i++)
        {
            m_adjacencyList[i] = new List<Vertex>();
        }
    }
}

public class Matching
{
    public Matching(Graph graph)
    {
        m_graph = graph;
        m_matchedVertex = new Vertex[graph.numVertices()];
        m_matchedVertex.AsSpan().Fill(-1);
        m_bridges = new VertexPair[graph.numVertices()];
        m_clearToken = 0;
        m_tree = new Node[graph.numVertices()];
        m_queue = new Queue<Vertex>();

        List<Vertex> unmatchedVertices = new();

        // Start with a greedy maximal matching
        for (Vertex v = 0; v < m_graph.numVertices(); ++v)
        {
            if (m_matchedVertex[v] == -1)
            {
                bool found = false;
                foreach (int w in m_graph.m_adjacencyList[v])
                {
                    if (m_matchedVertex[w] == -1)
                    {
                        match(v, w);
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    unmatchedVertices.Add(v);
                }
            }
        }

        List<Vertex> path = new();
        foreach (int v in unmatchedVertices)
        {
            if (m_matchedVertex[v] == -1)
            {
                if (findAugmentingPath(v, path))
                {
                    augment(CollectionsMarshal.AsSpan(path));
                    path.Clear();
                }
            }
        }
    }

    public Vertex getMatchedVertex(Vertex v)
    {
        return m_matchedVertex[v];
    }

    void match(Vertex v, Vertex w)
    {
        m_matchedVertex[v] = w;
        m_matchedVertex[w] = v;
    }

    void augment(ReadOnlySpan<Vertex> path)
    {
        for (int i = 0; i < path.Length; i += 2)
        {
            match(path[i], path[i + 1]);
        }
    }

    bool findAugmentingPath(Vertex root, List<Vertex> path)
    {
        // Clear out the forest
        m_clearToken++;

        // Start our tree root
        m_tree[root].m_depth = 0;
        m_tree[root].m_parent = -1;
        m_tree[root].m_clearToken = m_clearToken;
        m_tree[root].m_blossom = root;

        m_queue.Enqueue(root);

        while (m_queue.TryDequeue(out Vertex v))
        {
            foreach (int w in m_graph.m_adjacencyList[v])
            {
                if (examineEdge(root, v, w, path))
                {
                    m_queue.Clear();
                    return true;
                }
            }
        }

        return false;
    }

    bool examineEdge(Vertex root, Vertex v, Vertex w, List<Vertex> path)
    {
        Vertex vBar = find(v);
        Vertex wBar = find(w);

        if (vBar != wBar)
        {
            if (m_tree[wBar].m_clearToken != m_clearToken)
            {
                if (m_matchedVertex[w] == -1)
                {
                    buildAugmentingPath(root, v, w, path);
                    return true;
                }
                else
                {
                    extendTree(v, w);
                }
            }
            else if (m_tree[wBar].m_depth % 2 == 0)
            {
                shrinkBlossom(v, w);
            }
        }

        return false;
    }

    void buildAugmentingPath(Vertex root, Vertex v, Vertex w, List<Vertex> path)
    {
        path.Add(w);
        findPath(v, root, path);
    }

    void extendTree(Vertex v, Vertex w)
    {
        Vertex u = m_matchedVertex[w];

        ref Node nodeV = ref m_tree[v];
        ref Node nodeW = ref m_tree[w];
        ref Node nodeU = ref m_tree[u];

        nodeW.m_depth = nodeV.m_depth + 1 + (nodeV.m_depth & 1);    // Must be odd, so we add either 1 or 2
        nodeW.m_parent = v;
        nodeW.m_clearToken = m_clearToken;
        nodeW.m_blossom = w;

        nodeU.m_depth = nodeW.m_depth + 1;
        nodeU.m_parent = w;
        nodeU.m_clearToken = m_clearToken;
        nodeU.m_blossom = u;

        m_queue.Enqueue(u);
    }

    void shrinkBlossom(Vertex v, Vertex w)
    {
        Vertex b = findCommonAncestor(v, w);

        shrinkPath(b, v, w);
        shrinkPath(b, w, v);
    }

    void shrinkPath(Vertex b, Vertex v, Vertex w)
    {
        Vertex u = find(v);

        while (u != b)
        {
            makeUnion(b, u);
            Debug.Assert(m_matchedVertex[u] != -1);
            u = m_matchedVertex[u];
            makeUnion(b, u);
            makeRepresentative(b);
            m_queue.Enqueue(u);
            m_bridges[u] = new VertexPair(v, w);
            u = find(m_tree[u].m_parent);
        }
    }

    Vertex findCommonAncestor(Vertex v, Vertex w)
    {
        while (w != v)
        {
            if (m_tree[v].m_depth > m_tree[w].m_depth)
            {
                v = m_tree[v].m_parent;
            }
            else
            {
                w = m_tree[w].m_parent;
            }
        }

        return find(v);
    }

    void findPath(Vertex s, Vertex t, List<Vertex> path)
    {
        if (s == t)
        {
            path.Add(s);
        }
        else if (m_tree[s].m_depth % 2 == 0)
        {
            path.Add(s);
            path.Add(m_matchedVertex[s]);
            findPath(m_tree[m_matchedVertex[s]].m_parent, t, path);
        }
        else
        {
            Vertex v = (Vertex)m_bridges[s].first;
            Vertex w = (Vertex)m_bridges[s].second;

            path.Add(s);

            int offset = path.Count;
            findPath(v, m_matchedVertex[s], path);
            CollectionsMarshal.AsSpan(path).Slice(offset).Reverse();

            findPath(w, t, path);
        }
    }

    void makeUnion(int x, int y)
    {
        int xRoot = find(x);
        m_tree[xRoot].m_blossom = find(y);
    }

    void makeRepresentative(int x)
    {
        int xRoot = find(x);
        m_tree[xRoot].m_blossom = x;
        m_tree[x].m_blossom = x;
    }

    int find(int x)
    {
        if (m_tree[x].m_clearToken != m_clearToken)
        {
            return x;
        }

        if (x != m_tree[x].m_blossom)
        {
            // Path compression
            m_tree[x].m_blossom = find(m_tree[x].m_blossom);
        }

        return m_tree[x].m_blossom;
    }

    private int m_clearToken;

    private Graph m_graph;

    private Queue<Vertex> m_queue;
    private Vertex[] m_matchedVertex;

    private Node[] m_tree;

    private VertexPair[] m_bridges;

    private struct Node
    {
        public int m_depth;
        public Vertex m_parent;
        public Vertex m_blossom;

        public int m_clearToken;
    }
}

public struct VertexPair : IEquatable<VertexPair>
{
    public uint first;
    public uint second;

    public VertexPair(uint item1, uint item2)
    {
        this.first = item1;
        this.second = item2;
    }

    public VertexPair(int item1, int item2)
    {
        this.first = (uint)item1;
        this.second = (uint)item2;
    }

    public bool Equals(VertexPair other)
    {
        return
            first == other.first &&
            second == other.second;
    }

    public override int GetHashCode()
    {
        uint hashT = (uint)first.GetHashCode();
        uint hashU = (uint)second.GetHashCode();
        return (int)(hashT ^ (hashU + 0x9e3779b9 + (hashT << 6) + (hashT >> 2)));
    }
}

public static unsafe class QuadDecomposition
{
    private static bool canMergeTrianglesToQuad(Vector128<float> v0, Vector128<float> v1, Vector128<float> v2, Vector128<float> v3)
    {
        // Maximum distance of vertices from original plane in world space units
        float maximumDepthError = 0.5f;

        Vector128<float> n0 = normalize(normal(v0, v1, v2));
        Vector128<float> n2 = normalize(normal(v2, v3, v0));

        Vector128<float> planeDistA = Sse.AndNot(Vector128.Create(-0.0f), Sse41.DotProduct(n0, Sse.Subtract(v1, v3), 0x7F));
        Vector128<float> planeDistB = Sse.AndNot(Vector128.Create(-0.0f), Sse41.DotProduct(n2, Sse.Subtract(v1, v3), 0x7F));

        if (Sse.CompareScalarOrderedGreaterThan(planeDistA, Vector128.Create(maximumDepthError)) || Sse.CompareScalarOrderedGreaterThan(planeDistB, Vector128.Create(maximumDepthError)))
        {
            return false;
        }

        return true;
    }

    private static List<VertexPair> GetOrAdd(Dictionary<VertexPair, List<VertexPair>> dictionary, VertexPair key)
    {
        if (!dictionary.TryGetValue(key, out List<VertexPair>? list))
        {
            list = new List<VertexPair>();
            dictionary.Add(key, list);
        }
        return list;
    }

    public static List<uint> decompose(ReadOnlySpan<uint> indices, ReadOnlySpan<Vector128<float>> vertices)
    {
        int triangleCount = indices.Length / 3;

        Graph candidateGraph = new(triangleCount);

        Dictionary<VertexPair, List<VertexPair>> edgeToTriangle = new();

        uint* i = stackalloc uint[3];

        for (uint triangleIdx = 0; triangleIdx < triangleCount; ++triangleIdx)
        {
            i[0] = indices[(int)(3 * triangleIdx + 0)];
            i[1] = indices[(int)(3 * triangleIdx + 1)];
            i[2] = indices[(int)(3 * triangleIdx + 2)];

            GetOrAdd(edgeToTriangle, new VertexPair(i[0], i[1])).Add(new VertexPair(triangleIdx, i[2]));
            GetOrAdd(edgeToTriangle, new VertexPair(i[1], i[2])).Add(new VertexPair(triangleIdx, i[0]));
            GetOrAdd(edgeToTriangle, new VertexPair(i[2], i[0])).Add(new VertexPair(triangleIdx, i[1]));

            for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
            {
                VertexPair f = new(i[(edgeIdx + 1) % 3], i[edgeIdx]);
                if (!edgeToTriangle.TryGetValue(f, out List<VertexPair>? neighbors))
                {
                    continue;
                }

                foreach (VertexPair pair in neighbors)
                {
                    uint neighborTriangle = pair.first;
                    uint apex = pair.second;

                    int quad0 = (int)i[edgeIdx];
                    int quad1 = (int)apex;
                    int quad2 = (int)i[(edgeIdx + 1) % 3];
                    int quad3 = (int)i[(edgeIdx + 2) % 3];

                    if (canMergeTrianglesToQuad(vertices[quad0], vertices[quad1], vertices[quad2], vertices[quad3]))
                    {
                        candidateGraph.m_adjacencyList[triangleIdx].Add((int)neighborTriangle);
                        candidateGraph.m_adjacencyList[neighborTriangle].Add((int)triangleIdx);
                    }
                }
            }
        }


        uint quadCount = 0;

        Matching matching = new(candidateGraph);
        List<uint> result = new();

        for (uint triangleIdx = 0; triangleIdx < triangleCount; ++triangleIdx)
        {
            int neighbor = matching.getMatchedVertex((int)triangleIdx);

            // No quad found
            if (neighbor == -1)
            {
                var i0 = indices[(int)(3 * triangleIdx + 0)];
                var i1 = indices[(int)(3 * triangleIdx + 1)];
                var i2 = indices[(int)(3 * triangleIdx + 2)];

                result.Add(i0);
                result.Add(i2);
                result.Add(i1);
                result.Add(i0);
            }
            else if (triangleIdx < (uint)neighbor)
            {
                i[0] = indices[(int)(3 * triangleIdx + 0)];
                i[1] = indices[(int)(3 * triangleIdx + 1)];
                i[2] = indices[(int)(3 * triangleIdx + 2)];

                // Find out which edge was matched
                for (uint edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
                {
                    VertexPair f = new(i[(edgeIdx + 1) % 3], i[edgeIdx]);
                    if (!edgeToTriangle.TryGetValue(f, out List<VertexPair>? neighbors))
                    {
                        continue;
                    }

                    foreach (VertexPair pair in neighbors)
                    {
                        if (pair.first == neighbor)
                        {
                            result.Add(i[edgeIdx]);
                            result.Add(i[(edgeIdx + 2) % 3]);
                            result.Add(i[(edgeIdx + 1) % 3]);
                            result.Add(pair.second);

                            quadCount++;

                            goto nextTriangle;
                        }
                    }
                }
            }

        nextTriangle:
            continue;
        }

        return result;
    }
}