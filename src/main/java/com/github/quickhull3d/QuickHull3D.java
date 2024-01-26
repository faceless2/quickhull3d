package com.github.quickhull3d;

/*
 * #%L
 * A Robust 3D Convex Hull Algorithm in Java
 * %%
 * Copyright (C) 2004 - 2014 John E. Lloyd
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.StreamTokenizer;
import java.nio.charset.Charset;
import java.util.*;

/**
 * Computes the convex hull of a set of three dimensional points.
 * <p>
 * The algorithm is a three dimensional implementation of Quickhull, as
 * described in Barber, Dobkin, and Huhdanpaa, <a
 * href=http://citeseer.ist.psu.edu/barber96quickhull.html> ``The Quickhull
 * Algorithm for Convex Hulls''</a> (ACM Transactions on Mathematical Software,
 * Vol. 22, No. 4, December 1996), and has a complexity of O(n log(n)) with
 * respect to the number of points. A well-known C implementation of Quickhull
 * that works for arbitrary dimensions is provided by <a
 * href=http://www.qhull.org>qhull</a>.
 * <p>
 * A hull is constructed by providing a set of points to either a constructor or
 * a {@link #build(Point3d[]) build} method. After the hull is built, its
 * vertices and faces can be retrieved using {@link #getVertices() getVertices}
 * and {@link #getFaces() getFaces}. A typical usage might look like this:
 * 
 * <pre>
 * // x y z coordinates of 6 points
 * Point3d[] points = new Point3d[]{
 *     new Point3d(0.0, 0.0, 0.0),
 *     new Point3d(1.0, 0.5, 0.0),
 *     new Point3d(2.0, 0.0, 0.0),
 *     new Point3d(0.5, 0.5, 0.5),
 *     new Point3d(0.0, 0.0, 2.0),
 *     new Point3d(0.1, 0.2, 0.3),
 *     new Point3d(0.0, 2.0, 0.0),
 * };
 * 
 * QuickHull3D hull = new QuickHull3D();
 * hull.build(points);
 * 
 * System.out.println(&quot;Vertices:&quot;);
 * Point3d[] vertices = hull.getVertices();
 * for (int i = 0; i &lt; vertices.length; i++) {
 *     Point3d pnt = vertices[i];
 *     System.out.println(pnt.x + &quot; &quot; + pnt.y + &quot; &quot; + pnt.z);
 * }
 * 
 * System.out.println(&quot;Faces:&quot;);
 * int[][] faceIndices = hull.getFaces();
 * for (int i = 0; i &lt; faceIndices.length; i++) {
 *     for (int k = 0; k &lt; faceIndices[i].length; k++) {
 *         System.out.print(faceIndices[i][k] + &quot; &quot;);
 *     }
 *     System.out.println(&quot;&quot;);
 * }
 * </pre>
 * 
 * As a convenience, there are also {@link #build(double[]) build} and
 * {@link #getVertices(double[]) getVertex} methods which pass point information
 * using an array of doubles.
 * <h2><a id="distTol">Robustness</a></h2> Because this algorithm uses floating
 * point arithmetic, it is potentially vulnerable to errors arising from
 * numerical imprecision. We address this problem in the same way as <a
 * href=http://www.qhull.org>qhull</a>, by merging faces whose edges are not
 * clearly convex. A face is convex if its edges are convex, and an edge is
 * convex if the centroid of each adjacent plane is clearly <i>below</i> the
 * plane of the other face. The centroid is considered below a plane if its
 * distance to the plane is less than the negative of a
 * {@link #getDistanceTolerance() distance tolerance}. This tolerance represents
 * the smallest distance that can be reliably computed within the available
 * numeric precision. It is normally computed automatically from the point data,
 * although an application may {@link #setExplicitDistanceTolerance set this
 * tolerance explicitly}.
 * <p>
 * Numerical problems are more likely to arise in situations where data points
 * lie on or within the faces or edges of the convex hull. We have tested
 * QuickHull3D for such situations by computing the convex hull of a random
 * point set, then adding additional randomly chosen points which lie very close
 * to the hull vertices and edges, and computing the convex hull again. The hull
 * is deemed correct if {@link #check check} returns <code>true</code>. These
 * tests have been successful for a large number of trials and so we are
 * confident that QuickHull3D is reasonably robust.
 * <h3>Merged Faces</h3> The merging of faces means that the faces returned by
 * QuickHull3D may be convex polygons instead of triangles. If triangles are
 * desired, the application may {@link #triangulate triangulate} the faces, but
 * it should be noted that this may result in triangles which are very small or
 * thin and hence difficult to perform reliable convexity tests on. In other
 * words, triangulating a merged face is likely to restore the numerical
 * problems which the merging process removed. Hence is it possible that, after
 * triangulation, {@link #check check} will fail (the same behavior is observed
 * with triangulated output from <a href=http://www.qhull.org>qhull</a>).
 * <h3>Degenerate Input</h3>It is assumed that the input points are
 * non-degenerate in that they are not coincident, colinear, or colplanar, and
 * thus the convex hull has a non-zero volume. If the input points are detected
 * to be degenerate within the {@link #getDistanceTolerance() distance
 * tolerance}, an IllegalArgumentException will be thrown.
 * 
 * @author John E. Lloyd, Fall 2004
 */
public class QuickHull3D {

    /**
     * Specifies that (on output) vertex indices for a face should be listed in
     * clockwise order.
     */
    public static final int CLOCKWISE = 0x1;

    /**
     * Specifies that (on output) the vertex indices for a face should be
     * numbered starting from 1.
     */
    public static final int INDEXED_FROM_ONE = 0x2;

    /**
     * Specifies that (on output) the vertex indices for a face should be
     * numbered starting from 0.
     */
    public static final int INDEXED_FROM_ZERO = 0x4;

    /**
     * Specifies that (on output) the vertex indices for a face should be
     * numbered with respect to the original input points.
     */
    public static final int POINT_RELATIVE = 0x8;

    /**
     * Specifies that the distance tolerance should be computed automatically
     * from the input point data.
     */
    public static final double AUTOMATIC_TOLERANCE = -1;

    protected int findIndex = -1;

    // estimated size of the point set
    protected double charLength;

    protected Vertex[] pointBuffer = new Vertex[0];

    protected int[] vertexPointIndices = new int[0];

    private Face[] discardedFaces = new Face[3];

    private Vertex[] maxVtxs = new Vertex[3];

    private Vertex[] minVtxs = new Vertex[3];

    protected Vector<Face> faces = new Vector<Face>(16);

    protected Vector<HalfEdge> horizon = new Vector<HalfEdge>(16);

    private FaceList newFaces = new FaceList();

    private VertexList unclaimed = new VertexList();

    private VertexList claimed = new VertexList();

    protected int numVertices;

    protected int numFaces;

    protected int numPoints;

    protected double explicitTolerance = AUTOMATIC_TOLERANCE;

    protected double tolerance;

    /**
     * Precision of a double.
     */
    private static final double DOUBLE_PREC = 2.2204460492503131e-16;

    /**
     * Returns the distance tolerance that was used for the most recently
     * computed hull. The distance tolerance is used to determine when faces are
     * unambiguously convex with respect to each other, and when points are
     * unambiguously above or below a face plane, in the presence of <a
     * href=#distTol>numerical imprecision</a>. Normally, this tolerance is
     * computed automatically for each set of input points, but it can be set
     * explicitly by the application.
     * 
     * @return distance tolerance
     * @see QuickHull3D#setExplicitDistanceTolerance
     */
    public double getDistanceTolerance() {
        return tolerance;
    }

    /**
     * Sets an explicit distance tolerance for convexity tests. If
     * {@link #AUTOMATIC_TOLERANCE AUTOMATIC_TOLERANCE} is specified (the
     * default), then the tolerance will be computed automatically from the
     * point data.
     * 
     * @param tol
     *            explicit tolerance
     * @see #getDistanceTolerance
     */
    public void setExplicitDistanceTolerance(double tol) {
        explicitTolerance = tol;
    }

    /**
     * Returns the explicit distance tolerance.
     * 
     * @return explicit tolerance
     * @see #setExplicitDistanceTolerance
     */
    public double getExplicitDistanceTolerance() {
        return explicitTolerance;
    }

    private void addPointToFace(Vertex vtx, Face face) {
        vtx.face = face;

        if (face.outside == null) {
            claimed.add(vtx);
        } else {
            claimed.insertBefore(vtx, face.outside);
        }
        face.outside = vtx;
    }

    private void removePointFromFace(Vertex vtx, Face face) {
        if (vtx == face.outside) {
            if (vtx.next != null && vtx.next.face == face) {
                face.outside = vtx.next;
            } else {
                face.outside = null;
            }
        }
        claimed.delete(vtx);
    }

    private Vertex removeAllPointsFromFace(Face face) {
        if (face.outside != null) {
            Vertex end = face.outside;
            while (end.next != null && end.next.face == face) {
                end = end.next;
            }
            claimed.delete(face.outside, end);
            end.next = null;
            return face.outside;
        } else {
            return null;
        }
    }

    /**
     * Creates an empty convex hull object.
     */
    public QuickHull3D() {
    }

    /**
     * Creates a convex hull object and initializes it to the convex hull of a
     * set of points whose coordinates are given by an array of doubles.
     * 
     * @param coords
     *            x, y, and z coordinates of each input point. The length of
     *            this array will be three times the the number of input points.
     * @throws IllegalArgumentException
     *             the number of input points is less than four, or the points
     *             appear to be coincident, colinear, or coplanar.
     */
    public QuickHull3D(double[] coords) {
        build(coords, coords.length / 3);
    }

    /**
     * Creates a convex hull object and initializes it to the convex hull of a
     * set of points.
     * 
     * @param points
     *            input points.
     * @throws IllegalArgumentException
     *             the number of input points is less than four, or the points
     *             appear to be coincident, colinear, or coplanar.
     */
    public QuickHull3D(Point3d[] points) {
        build(points, points.length);
    }

    private HalfEdge findHalfEdge(Vertex tail, Vertex head) {
        // brute force ... OK, since setHull is not used much
        for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
            HalfEdge he = it.next().findEdge(tail, head);
            if (he != null) {
                return he;
            }
        }
        return null;
    }

    protected void setHull(double[] coords, int nump, int[][] faceIndices, int numf) {
        setPoints(coords, nump);
        computeMaxAndMin();
        for (int i = 0; i < numf; i++) {
            Face face = Face.create(pointBuffer, faceIndices[i]);
            HalfEdge he = face.he0;
            do {
                HalfEdge heOpp = findHalfEdge(he.head(), he.tail());
                if (heOpp != null) {
                    he.setOpposite(heOpp);
                }
                he = he.next;
            } while (he != face.he0);
            faces.add(face);
        }
    }

    private void printQhullErrors(Process proc) throws IOException {
        boolean wrote = false;
        InputStream es = proc.getErrorStream();
        StringBuffer error = new StringBuffer();
        while (es.available() > 0) {
            error.append((char) es.read());
            wrote = true;
        }
        if (wrote) {
            error.append(" ");
            error(error.toString());
        }
    }

    protected void setFromQhull(double[] coords, int nump, boolean triangulate) {
        String commandStr = "./qhull i";
        if (triangulate) {
            commandStr += " -Qt";
        }
        try {
            Process proc = Runtime.getRuntime().exec(commandStr);
            PrintStream ps = new PrintStream(proc.getOutputStream(), false, Charset.defaultCharset().name());
            StreamTokenizer stok = new StreamTokenizer(new InputStreamReader(proc.getInputStream(), Charset.defaultCharset()));

            ps.println("3 " + nump);
            for (int i = 0; i < nump; i++) {
                ps.println(coords[i * 3 + 0] + " " + coords[i * 3 + 1] + " " + coords[i * 3 + 2]);
            }
            ps.flush();
            ps.close();
            Vector<Integer> indexList = new Vector<Integer>(3);
            stok.eolIsSignificant(true);
            printQhullErrors(proc);

            do {
                stok.nextToken();
            } while (stok.sval == null || !stok.sval.startsWith("MERGEexact"));
            for (int i = 0; i < 4; i++) {
                stok.nextToken();
            }
            if (stok.ttype != StreamTokenizer.TT_NUMBER) {
                throw new IllegalArgumentException("Expecting number of faces");
            }
            int numf = (int) stok.nval;
            stok.nextToken(); // clear EOL
            int[][] faceIndices = new int[numf][];
            for (int i = 0; i < numf; i++) {
                indexList.clear();
                while (stok.nextToken() != StreamTokenizer.TT_EOL) {
                    if (stok.ttype != StreamTokenizer.TT_NUMBER) {
                        throw new IllegalArgumentException("Expecting face index");
                    }
                    indexList.add(0, Integer.valueOf((int) stok.nval));
                }
                faceIndices[i] = new int[indexList.size()];
                int k = 0;
                for (Iterator<Integer> it = indexList.iterator(); it.hasNext();) {
                    faceIndices[i][k++] = it.next().intValue();
                }
            }
            setHull(coords, nump, faceIndices, numf);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("problem during hull calculation", e);
        }
    }

    /**
     * print all points to the print stream (very point a line)
     * 
     * @param ps
     *            the print stream to write to
     */
    public void printPoints(PrintStream ps) {
        for (int i = 0; i < numPoints; i++) {
            Point3d pnt = pointBuffer[i].pnt;
            ps.println(pnt.x + ", " + pnt.y + ", " + pnt.z + ",");
        }
    }

    /**
     * Constructs the convex hull of a set of points whose coordinates are given
     * by an array of doubles.
     * 
     * @param coords
     *            x, y, and z coordinates of each input point. The length of
     *            this array will be three times the number of input points.
     * @throws IllegalArgumentException
     *             the number of input points is less than four, or the points
     *             appear to be coincident, colinear, or coplanar.
     */
    public void build(double[] coords) {
        build(coords, coords.length / 3);
    }

    /**
     * Constructs the convex hull of a set of points whose coordinates are given
     * by an array of doubles.
     * 
     * @param coords
     *            x, y, and z coordinates of each input point. The length of
     *            this array must be at least three times <code>nump</code>.
     * @param nump
     *            number of input points
     * @throws IllegalArgumentException
     *             the number of input points is less than four or greater than
     *             1/3 the length of <code>coords</code>, or the points appear
     *             to be coincident, colinear, or coplanar.
     */
    public void build(double[] coords, int nump) throws IllegalArgumentException {
        if (nump < 4) {
            throw new IllegalArgumentException("Less than four input points specified");
        }
        if (coords.length / 3 < nump) {
            throw new IllegalArgumentException("Coordinate array too small for specified number of points");
        }
        setPoints(coords, nump);
        buildHull();
    }

    /**
     * Constructs the convex hull of a set of points.
     * 
     * @param points
     *            input points
     * @throws IllegalArgumentException
     *             the number of input points is less than four, or the points
     *             appear to be coincident, colinear, or coplanar.
     */
    public void build(Point3d[] points) throws IllegalArgumentException {
        build(points, points.length);
    }

    /**
     * Constructs the convex hull of a set of points.
     * 
     * @param points
     *            input points
     * @param nump
     *            number of input points
     * @throws IllegalArgumentException
     *             the number of input points is less than four or greater then
     *             the length of <code>points</code>, or the points appear to be
     *             coincident, colinear, or coplanar.
     */
    public void build(Point3d[] points, int nump) {
        if (nump < 4) {
            throw new IllegalArgumentException("Less than four input points specified");
        }
        if (points.length < nump) {
            throw new IllegalArgumentException("Point array too small for specified number of points");
        }
        setPoints(Arrays.asList(points));
        buildHull();
    }

    /**
     * Constructs the convex hull of a set of points
     * 
     * @param points
     *            input points
     */
    public void build(List<Point3d> points) {
        if (points.size() < 4) {
            throw new IllegalArgumentException("Less than four input points specified");
        }
        setPoints(points);
        buildHull();
    }

    /**
     * Triangulates any non-triangular hull faces. In some cases, due to
     * precision issues, the resulting triangles may be very thin or small, and
     * hence appear to be non-convex (this same limitation is present in <a
     * href=http://www.qhull.org>qhull</a>).
     */
    public void triangulate() {
        double minArea = 1000 * charLength * DOUBLE_PREC;
        newFaces.clear();
        for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
            Face face = it.next();
            if (face.mark == Face.VISIBLE) {
                face.triangulate(newFaces, minArea);
                // splitFace (face);
            }
        }
        for (Face face = newFaces.first(); face != null; face = face.next) {
            faces.add(face);
        }
    }

    // private void splitFace (Face face)
    // {
    // Face newFace = face.split();
    // if (newFace != null)
    // { newFaces.add (newFace);
    // splitFace (newFace);
    // splitFace (face);
    // }
    // }

    private void setPoints(double[] coords, int len) {
        List<Point3d> points = new ArrayList<Point3d>(len);
        for (int i = 0; i < len * 3;) {
            points.add(new Point3d(coords[i++], coords[i++], coords[i++]));
        }
        setPoints(points);
    }

    private void setPoints(List<Point3d> pnts) {
        int nump = pnts.size();
        vertexPointIndices = new int[nump];
        pointBuffer = new Vertex[nump];
        for (int i = 0; i < pnts.size(); i++) {
            pointBuffer[i] = new Vertex(pnts.get(i));
            pointBuffer[i].index = i;
        }
        faces.clear();
        claimed.clear();
        numFaces = 0;
        numPoints = nump;
    }

    private void computeMaxAndMin() {
        Vector3d max = new Vector3d();
        Vector3d min = new Vector3d();

        for (int i = 0; i < 3; i++) {
            maxVtxs[i] = minVtxs[i] = pointBuffer[0];
        }
        max.set(pointBuffer[0].pnt);
        min.set(pointBuffer[0].pnt);

        for (int i = 1; i < numPoints; i++) {
            Point3d pnt = pointBuffer[i].pnt;
            if (pnt.x > max.x) {
                max.x = pnt.x;
                maxVtxs[0] = pointBuffer[i];
            } else if (pnt.x < min.x) {
                min.x = pnt.x;
                minVtxs[0] = pointBuffer[i];
            }
            if (pnt.y > max.y) {
                max.y = pnt.y;
                maxVtxs[1] = pointBuffer[i];
            } else if (pnt.y < min.y) {
                min.y = pnt.y;
                minVtxs[1] = pointBuffer[i];
            }
            if (pnt.z > max.z) {
                max.z = pnt.z;
                maxVtxs[2] = pointBuffer[i];
            } else if (pnt.z < min.z) {
                min.z = pnt.z;
                minVtxs[2] = pointBuffer[i];
            }
        }

        // this epsilon formula comes from QuickHull, and I'm
        // not about to quibble.
        charLength = Math.max(max.x - min.x, max.y - min.y);
        charLength = Math.max(max.z - min.z, charLength);
        if (explicitTolerance == AUTOMATIC_TOLERANCE) {
            tolerance =
                    3 * DOUBLE_PREC * (Math.max(Math.abs(max.x), Math.abs(min.x)) + Math.max(Math.abs(max.y), Math.abs(min.y)) + Math.max(Math.abs(max.z), Math.abs(min.z)));
        } else {
            tolerance = explicitTolerance;
        }
    }

    /**
     * Creates the initial simplex from which the hull will be built.
     */
    private void createInitialSimplex() throws IllegalArgumentException {
        double max = 0;
        int imax = 0;

        for (int i = 0; i < 3; i++) {
            double diff = maxVtxs[i].pnt.get(i) - minVtxs[i].pnt.get(i);
            if (diff > max) {
                max = diff;
                imax = i;
            }
        }

        if (max <= tolerance) {
            throw new IllegalArgumentException("Input points appear to be coincident");
        }
        Vertex[] vtx = new Vertex[4];
        // set first two vertices to be those with the greatest
        // one dimensional separation

        vtx[0] = maxVtxs[imax];
        vtx[1] = minVtxs[imax];

        // set third vertex to be the vertex farthest from
        // the line between vtx0 and vtx1
        Vector3d diff02 = new Vector3d();
        Vector3d nrml = new Vector3d();
        Vector3d xprod = new Vector3d();
        double maxSqr = 0;
        Vector3d u01 = new Vector3d(vtx[1].pnt).sub(vtx[0].pnt).normalize();
        for (int i = 0; i < numPoints; i++) {
            diff02.set(pointBuffer[i].pnt).sub(vtx[0].pnt);
            xprod.set(u01).cross(diff02);
            double lenSqr = xprod.normSquared();
            if (lenSqr > maxSqr && pointBuffer[i] != vtx[0] && pointBuffer[i] != vtx[1]) {
                maxSqr = lenSqr;
                vtx[2] = pointBuffer[i];
                nrml.set(xprod);
            }
        }
        if (Math.sqrt(maxSqr) <= 100 * tolerance) {
            throw new IllegalArgumentException("Input points appear to be colinear");
        }
        nrml.normalize();

        double maxDist = 0;
        double d0 = vtx[2].pnt.dot(nrml);
        for (int i = 0; i < numPoints; i++) {
            double dist = Math.abs(pointBuffer[i].pnt.dot(nrml) - d0);
            if (dist > maxDist && pointBuffer[i] != vtx[0] && // paranoid
                    pointBuffer[i] != vtx[1] && pointBuffer[i] != vtx[2]) {
                maxDist = dist;
                vtx[3] = pointBuffer[i];
            }
        }
        if (Math.abs(maxDist) <= 100 * tolerance) {
            throw new IllegalArgumentException("Input points appear to be coplanar");
        }

        if (isDebugEnabled()) {
            debug("initial vertices:");
            debug(vtx[0].index + ": " + vtx[0].pnt);
            debug(vtx[1].index + ": " + vtx[1].pnt);
            debug(vtx[2].index + ": " + vtx[2].pnt);
            debug(vtx[3].index + ": " + vtx[3].pnt);
        }

        Face[] tris = new Face[4];

        if (vtx[3].pnt.dot(nrml) - d0 < 0) {
            tris[0] = Face.createTriangle(vtx[0], vtx[1], vtx[2]);
            tris[1] = Face.createTriangle(vtx[3], vtx[1], vtx[0]);
            tris[2] = Face.createTriangle(vtx[3], vtx[2], vtx[1]);
            tris[3] = Face.createTriangle(vtx[3], vtx[0], vtx[2]);

            for (int i = 0; i < 3; i++) {
                int k = (i + 1) % 3;
                tris[i + 1].getEdge(1).setOpposite(tris[k + 1].getEdge(0));
                tris[i + 1].getEdge(2).setOpposite(tris[0].getEdge(k));
            }
        } else {
            tris[0] = Face.createTriangle(vtx[0], vtx[2], vtx[1]);
            tris[1] = Face.createTriangle(vtx[3], vtx[0], vtx[1]);
            tris[2] = Face.createTriangle(vtx[3], vtx[1], vtx[2]);
            tris[3] = Face.createTriangle(vtx[3], vtx[2], vtx[0]);

            for (int i = 0; i < 3; i++) {
                int k = (i + 1) % 3;
                tris[i + 1].getEdge(0).setOpposite(tris[k + 1].getEdge(1));
                tris[i + 1].getEdge(2).setOpposite(tris[0].getEdge((3 - i) % 3));
            }
        }

        for (int i = 0; i < 4; i++) {
            faces.add(tris[i]);
        }

        for (int i = 0; i < numPoints; i++) {
            Vertex v = pointBuffer[i];

            if (v == vtx[0] || v == vtx[1] || v == vtx[2] || v == vtx[3]) {
                continue;
            }

            maxDist = tolerance;
            Face maxFace = null;
            for (int k = 0; k < 4; k++) {
                double dist = tris[k].distanceToPlane(v.pnt);
                if (dist > maxDist) {
                    maxFace = tris[k];
                    maxDist = dist;
                }
            }
            if (maxFace != null) {
                addPointToFace(v, maxFace);
            }
        }
    }

    /**
     * Returns the number of vertices in this hull.
     * 
     * @return number of vertices
     */
    public int getNumVertices() {
        return numVertices;
    }

    /**
     * Returns the vertex points in this hull.
     * 
     * @return array of vertex points
     * @see QuickHull3D#getVertices(double[])
     * @see QuickHull3D#getFaces()
     */
    public Point3d[] getVertices() {
        Point3d[] vtxs = new Point3d[numVertices];
        for (int i = 0; i < numVertices; i++) {
            vtxs[i] = pointBuffer[vertexPointIndices[i]].pnt;
        }
        return vtxs;
    }

    /**
     * Returns the coordinates of the vertex points of this hull.
     * 
     * @param coords
     *            returns the x, y, z coordinates of each vertex. This length of
     *            this array must be at least three times the number of
     *            vertices.
     * @return the number of vertices
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getFaces()
     */
    public int getVertices(double[] coords) {
        for (int i = 0; i < numVertices; i++) {
            Point3d pnt = pointBuffer[vertexPointIndices[i]].pnt;
            coords[i * 3 + 0] = pnt.x;
            coords[i * 3 + 1] = pnt.y;
            coords[i * 3 + 2] = pnt.z;
        }
        return numVertices;
    }

    /**
     * Returns an array specifing the index of each hull vertex with respect to
     * the original input points.
     * 
     * @return vertex indices with respect to the original points
     */
    public int[] getVertexPointIndices() {
        int[] indices = new int[numVertices];
        for (int i = 0; i < numVertices; i++) {
            indices[i] = vertexPointIndices[i];
        }
        return indices;
    }

    /**
     * Returns the number of faces in this hull.
     * 
     * @return number of faces
     */
    public int getNumFaces() {
        return faces.size();
    }

    /**
     * Returns the faces associated with this hull.
     * <p>
     * Each face is represented by an integer array which gives the indices of
     * the vertices. These indices are numbered relative to the hull vertices,
     * are zero-based, and are arranged counter-clockwise. More control over the
     * index format can be obtained using {@link #getFaces(int)
     * getFaces(indexFlags)}.
     * 
     * @return array of integer arrays, giving the vertex indices for each face.
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getFaces(int)
     */
    public int[][] getFaces() {
        return getFaces(0);
    }

    /**
     * Returns the faces associated with this hull.
     * <p>
     * Each face is represented by an integer array which gives the indices of
     * the vertices. By default, these indices are numbered with respect to the
     * hull vertices (as opposed to the input points), are zero-based, and are
     * arranged counter-clockwise. However, this can be changed by setting
     * {@link #POINT_RELATIVE POINT_RELATIVE}, {@link #INDEXED_FROM_ONE
     * INDEXED_FROM_ONE}, or {@link #CLOCKWISE CLOCKWISE} in the indexFlags
     * parameter.
     * 
     * @param indexFlags
     *            specifies index characteristics (0 results in the default)
     * @return array of integer arrays, giving the vertex indices for each face.
     * @see QuickHull3D#getVertices()
     */
    public int[][] getFaces(int indexFlags) {
        int[][] allFaces = new int[faces.size()][];
        int k = 0;
        for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
            Face face = it.next();
            allFaces[k] = new int[face.numVertices()];
            getFaceIndices(allFaces[k], face, indexFlags);
            k++;
        }
        return allFaces;
    }

    /**
     * Prints the vertices and faces of this hull to the stream ps.
     * <p>
     * This is done using the Alias Wavefront .obj file format, with the
     * vertices printed first (each preceding by the letter <code>v</code>),
     * followed by the vertex indices for each face (each preceded by the letter
     * <code>f</code>).
     * <p>
     * The face indices are numbered with respect to the hull vertices (as
     * opposed to the input points), with a lowest index of 1, and are arranged
     * counter-clockwise. More control over the index format can be obtained
     * using {@link #print(PrintStream,int) print(ps,indexFlags)}.
     * 
     * @param ps
     *            stream used for printing
     * @see QuickHull3D#print(PrintStream,int)
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getFaces()
     */
    public void print(PrintStream ps) {
        print(ps, 0);
    }

    /**
     * Prints the vertices and faces of this hull to the stream ps.
     * <p>
     * This is done using the Alias Wavefront .obj file format, with the
     * vertices printed first (each preceding by the letter <code>v</code>),
     * followed by the vertex indices for each face (each preceded by the letter
     * <code>f</code>).
     * <p>
     * By default, the face indices are numbered with respect to the hull
     * vertices (as opposed to the input points), with a lowest index of 1, and
     * are arranged counter-clockwise. However, this can be changed by setting
     * {@link #POINT_RELATIVE POINT_RELATIVE}, {@link #INDEXED_FROM_ONE
     * INDEXED_FROM_ZERO}, or {@link #CLOCKWISE CLOCKWISE} in the indexFlags
     * parameter.
     * 
     * @param ps
     *            stream used for printing
     * @param indexFlags
     *            specifies index characteristics (0 results in the default).
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getFaces()
     */
    public void print(PrintStream ps, int indexFlags) {
        if ((indexFlags & INDEXED_FROM_ZERO) == 0) {
            indexFlags |= INDEXED_FROM_ONE;
        }
        for (int i = 0; i < numVertices; i++) {
            Point3d pnt = pointBuffer[vertexPointIndices[i]].pnt;
            ps.println("v " + pnt.x + " " + pnt.y + " " + pnt.z);
        }
        for (Iterator<Face> fi = faces.iterator(); fi.hasNext();) {
            Face face = fi.next();
            int[] indices = new int[face.numVertices()];
            getFaceIndices(indices, face, indexFlags);

            ps.print("f");
            for (int k = 0; k < indices.length; k++) {
                ps.print(" " + indices[k]);
            }
            ps.println("");
        }
    }

    private void getFaceIndices(int[] indices, Face face, int flags) {
        boolean ccw = (flags & CLOCKWISE) == 0;
        boolean indexedFromOne = (flags & INDEXED_FROM_ONE) != 0;
        boolean pointRelative = (flags & POINT_RELATIVE) != 0;

        HalfEdge hedge = face.he0;
        int k = 0;
        do {
            int idx = hedge.head().index;
            if (pointRelative) {
                idx = vertexPointIndices[idx];
            }
            if (indexedFromOne) {
                idx++;
            }
            indices[k++] = idx;
            hedge = (ccw ? hedge.next : hedge.prev);
        } while (hedge != face.he0);
    }

    protected void resolveUnclaimedPoints(FaceList newFaces) {
        Vertex vtxNext = unclaimed.first();
        for (Vertex vtx = vtxNext; vtx != null; vtx = vtxNext) {
            vtxNext = vtx.next;

            double maxDist = tolerance;
            Face maxFace = null;
            for (Face newFace = newFaces.first(); newFace != null; newFace = newFace.next) {
                if (newFace.mark == Face.VISIBLE) {
                    double dist = newFace.distanceToPlane(vtx.pnt);
                    if (dist > maxDist) {
                        maxDist = dist;
                        maxFace = newFace;
                    }
                    if (maxDist > 1000 * tolerance) {
                        break;
                    }
                }
            }
            if (maxFace != null) {
                addPointToFace(vtx, maxFace);
                if (isDebugEnabled() && vtx.index == findIndex) {
                    debug(findIndex + " CLAIMED BY " + maxFace.getVertexString());
                }
            } else {
                if (isDebugEnabled() && vtx.index == findIndex) {
                    debug(findIndex + " DISCARDED");
                }
            }
        }
    }

    private void deleteFacePoints(Face face, Face absorbingFace) {
        Vertex faceVtxs = removeAllPointsFromFace(face);
        if (faceVtxs != null) {
            if (absorbingFace == null) {
                unclaimed.addAll(faceVtxs);
            } else {
                Vertex vtxNext = faceVtxs;
                for (Vertex vtx = vtxNext; vtx != null; vtx = vtxNext) {
                    vtxNext = vtx.next;
                    double dist = absorbingFace.distanceToPlane(vtx.pnt);
                    if (dist > tolerance) {
                        addPointToFace(vtx, absorbingFace);
                    } else {
                        unclaimed.add(vtx);
                    }
                }
            }
        }
    }

    private static final int NONCONVEX_WRT_LARGER_FACE = 1;

    private static final int NONCONVEX = 2;

    private double oppFaceDistance(HalfEdge he) {
        return he.face.distanceToPlane(he.opposite.face.getCentroid());
    }

    private boolean doAdjacentMerge(Face face, int mergeType) {
        HalfEdge hedge = face.he0;

        boolean convex = true;
        do {
            Face oppFace = hedge.oppositeFace();
            boolean merge = false;

            if (mergeType == NONCONVEX) { // then merge faces if they are
                                          // definitively non-convex
                if (oppFaceDistance(hedge) > -tolerance || oppFaceDistance(hedge.opposite) > -tolerance) {
                    merge = true;
                }
            } else {
                // mergeType == NONCONVEX_WRT_LARGER_FACE
                // merge faces if they are parallel or non-convex
                // wrt to the larger face; otherwise, just mark
                // the face non-convex for the second pass.
                if (face.area > oppFace.area) {
                    if (oppFaceDistance(hedge) > -tolerance) {
                        merge = true;
                    } else if (oppFaceDistance(hedge.opposite) > -tolerance) {
                        convex = false;
                    }
                } else {
                    if (oppFaceDistance(hedge.opposite) > -tolerance) {
                        merge = true;
                    } else if (oppFaceDistance(hedge) > -tolerance) {
                        convex = false;
                    }
                }
            }

            if (merge) {
                if (isDebugEnabled()) {
                    debug("  merging " + face.getVertexString() + "  and  " + oppFace.getVertexString());
                }

                int numd = face.mergeAdjacentFace(hedge, discardedFaces);
                for (int i = 0; i < numd; i++) {
                    deleteFacePoints(discardedFaces[i], face);
                }
                if (isDebugEnabled()) {
                    debug("  result: " + face.getVertexString());
                }
                return true;
            }
            hedge = hedge.next;
        } while (hedge != face.he0);
        if (!convex) {
            face.mark = Face.NON_CONVEX;
        }
        return false;
    }

    private void calculateHorizon(Point3d eyePnt, HalfEdge edge0, Face face, Vector<HalfEdge> horizon) {
        // oldFaces.add (face);
        deleteFacePoints(face, null);
        face.mark = Face.DELETED;
        if (isDebugEnabled()) {
            debug("  visiting face " + face.getVertexString());
        }
        HalfEdge edge;
        if (edge0 == null) {
            edge0 = face.getEdge(0);
            edge = edge0;
        } else {
            edge = edge0.getNext();
        }
        do {
            Face oppFace = edge.oppositeFace();
            if (oppFace.mark == Face.VISIBLE) {
                if (oppFace.distanceToPlane(eyePnt) > tolerance) {
                    calculateHorizon(eyePnt, edge.getOpposite(), oppFace, horizon);
                } else {
                    horizon.add(edge);
                    if (isDebugEnabled()) {
                        debug("  adding horizon edge " + edge.getVertexString());
                    }
                }
            }
            edge = edge.getNext();
        } while (edge != edge0);
    }

    private HalfEdge addAdjoiningFace(Vertex eyeVtx, HalfEdge he) {
        Face face = Face.createTriangle(eyeVtx, he.tail(), he.head());
        faces.add(face);
        face.getEdge(-1).setOpposite(he.getOpposite());
        return face.getEdge(0);
    }

    private void addNewFaces(FaceList newFaces, Vertex eyeVtx, Vector<HalfEdge> horizon) {
        newFaces.clear();

        HalfEdge hedgeSidePrev = null;
        HalfEdge hedgeSideBegin = null;

        for (Iterator<HalfEdge> it = horizon.iterator(); it.hasNext();) {
            HalfEdge horizonHe = it.next();
            HalfEdge hedgeSide = addAdjoiningFace(eyeVtx, horizonHe);
            if (isDebugEnabled()) {
                debug("new face: " + hedgeSide.face.getVertexString());
            }
            if (hedgeSidePrev != null) {
                hedgeSide.next.setOpposite(hedgeSidePrev);
            } else {
                hedgeSideBegin = hedgeSide;
            }
            newFaces.add(hedgeSide.getFace());
            hedgeSidePrev = hedgeSide;
        }
        hedgeSideBegin.next.setOpposite(hedgeSidePrev);
    }

    private Vertex nextPointToAdd() {
        if (!claimed.isEmpty()) {
            Face eyeFace = claimed.first().face;
            Vertex eyeVtx = null;
            double maxDist = 0;
            for (Vertex vtx = eyeFace.outside; vtx != null && vtx.face == eyeFace; vtx = vtx.next) {
                double dist = eyeFace.distanceToPlane(vtx.pnt);
                if (dist > maxDist) {
                    maxDist = dist;
                    eyeVtx = vtx;
                }
            }
            return eyeVtx;
        } else {
            return null;
        }
    }

    private void addPointToHull(Vertex eyeVtx) {
        horizon.clear();
        unclaimed.clear();

        if (isDebugEnabled()) {
            debug("Adding point: " + eyeVtx.index);
            debug(" which is " + eyeVtx.face.distanceToPlane(eyeVtx.pnt) + " above face " + eyeVtx.face.getVertexString());
        }
        removePointFromFace(eyeVtx, eyeVtx.face);
        calculateHorizon(eyeVtx.pnt, null, eyeVtx.face, horizon);
        newFaces.clear();
        addNewFaces(newFaces, eyeVtx, horizon);

        // first merge pass ... merge faces which are non-convex
        // as determined by the larger face

        for (Face face = newFaces.first(); face != null; face = face.next) {
            if (face.mark == Face.VISIBLE) {
                while (doAdjacentMerge(face, NONCONVEX_WRT_LARGER_FACE))
                    ;
            }
        }
        // second merge pass ... merge faces which are non-convex
        // wrt either face
        for (Face face = newFaces.first(); face != null; face = face.next) {
            if (face.mark == Face.NON_CONVEX) {
                face.mark = Face.VISIBLE;
                while (doAdjacentMerge(face, NONCONVEX))
                    ;
            }
        }
        resolveUnclaimedPoints(newFaces);
    }

    private void buildHull() {
        int cnt = 0;
        Vertex eyeVtx;

        computeMaxAndMin();
        createInitialSimplex();
        while ((eyeVtx = nextPointToAdd()) != null) {
            addPointToHull(eyeVtx);
            cnt++;
            debug("iteration " + cnt + " done");
        }
        reindexFacesAndVertices();
        debug("hull done");
    }

    private void markFaceVertices(Face face, int mark) {
        HalfEdge he0 = face.getFirstEdge();
        HalfEdge he = he0;
        do {
            he.head().index = mark;
            he = he.next;
        } while (he != he0);
    }

    private void reindexFacesAndVertices() {
        for (int i = 0; i < numPoints; i++) {
            pointBuffer[i].index = -1;
        }
        // remove inactive faces and mark active vertices
        numFaces = 0;
        for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
            Face face = it.next();
            if (face.mark != Face.VISIBLE) {
                it.remove();
            } else {
                markFaceVertices(face, 0);
                numFaces++;
            }
        }
        // reindex vertices
        numVertices = 0;
        for (int i = 0; i < numPoints; i++) {
            Vertex vtx = pointBuffer[i];
            if (vtx.index == 0) {
                vertexPointIndices[numVertices] = i;
                vtx.index = numVertices++;
            }
        }
    }

    private boolean checkFaceConvexity(Face face, double tol, PrintStream ps) {
        double dist;
        HalfEdge he = face.he0;
        do {
            face.checkConsistency();
            // make sure edge is convex
            dist = oppFaceDistance(he);
            if (dist > tol) {
                if (ps != null) {
                    ps.println("Edge " + he.getVertexString() + " non-convex by " + dist);
                }
                return false;
            }
            dist = oppFaceDistance(he.opposite);
            if (dist > tol) {
                if (ps != null) {
                    ps.println("Opposite edge " + he.opposite.getVertexString() + " non-convex by " + dist);
                }
                return false;
            }
            if (he.next.oppositeFace() == he.oppositeFace()) {
                if (ps != null) {
                    ps.println("Redundant vertex " + he.head().index + " in face " + face.getVertexString());
                }
                return false;
            }
            he = he.next;
        } while (he != face.he0);
        return true;
    }

    private boolean checkFaces(double tol, PrintStream ps) {
        // check edge convexity
        boolean convex = true;
        for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
            Face face = it.next();
            if (face.mark == Face.VISIBLE && !checkFaceConvexity(face, tol, ps)) {
                convex = false;
            }
        }
        return convex;
    }

    /**
     * Checks the correctness of the hull using the distance tolerance returned
     * by {@link QuickHull3D#getDistanceTolerance getDistanceTolerance}; see
     * {@link QuickHull3D#check(PrintStream,double) check(PrintStream,double)}
     * for details.
     * 
     * @param ps
     *            print stream for diagnostic messages; may be set to
     *            <code>null</code> if no messages are desired.
     * @return true if the hull is valid
     * @see QuickHull3D#check(PrintStream,double)
     */
    public boolean check(PrintStream ps) {
        return check(ps, getDistanceTolerance());
    }

    /**
     * Checks the correctness of the hull. This is done by making sure that no
     * faces are non-convex and that no points are outside any face. These tests
     * are performed using the distance tolerance <i>tol</i>. Faces are
     * considered non-convex if any edge is non-convex, and an edge is
     * non-convex if the centroid of either adjoining face is more than
     * <i>tol</i> above the plane of the other face. Similarly, a point is
     * considered outside a face if its distance to that face's plane is more
     * than 10 times <i>tol</i>.
     * <p>
     * If the hull has been {@link #triangulate triangulated}, then this routine
     * may fail if some of the resulting triangles are very small or thin.
     * 
     * @param ps
     *            print stream for diagnostic messages; may be set to
     *            <code>null</code> if no messages are desired.
     * @param tol
     *            distance tolerance
     * @return true if the hull is valid
     * @see QuickHull3D#check(PrintStream)
     */
    public boolean check(PrintStream ps, double tol) {
        // check to make sure all edges are fully connected
        // and that the edges are convex
        double dist;
        double pointTol = 10 * tol;

        if (!checkFaces(tolerance, ps)) {
            return false;
        }

        // check point inclusion

        for (int i = 0; i < numPoints; i++) {
            Point3d pnt = pointBuffer[i].pnt;
            for (Iterator<Face> it = faces.iterator(); it.hasNext();) {
                Face face = it.next();
                if (face.mark == Face.VISIBLE) {
                    dist = face.distanceToPlane(pnt);
                    if (dist > pointTol) {
                        if (ps != null) {
                            ps.println("Point " + i + " " + dist + " above face " + face.getVertexString());
                        }
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Modify the Convex hull by selecting concave points that do not deflect a
     * vector by more then "threshold" degrees. Requires all faces are triangles
     * 
     * @param threshold
     *            the maximum value in degrees that an edge can deviate
     */
    public void concave(final float threshold) {
        // Based on the idea in
        // "A New Concave Hull Algorithm and Concaveness Measure for n-dimensional Datasets*",
        // 2010
        // by JIN-SEO PARK AND SE-JONG OH, except modifying the threshold test
        // to check for angle rather than distance.

        // 1 Find the set of edges - put an arbitrary one of each pair of
        // half-pairs into "edges"
        Set<HalfEdge> seen = new HashSet<HalfEdge>();
        List<HalfEdge> edges = new ArrayList<HalfEdge>();
        for (Face face : faces) {
            HalfEdge h = face.he0;
            do {
                if (!seen.contains(h)) {
                    edges.add(h);
                    seen.add(h);
                    if (h.opposite != null) {
                        seen.add(h.opposite);
                    }
                }
                h = h.next;
            } while (h != face.he0);
        }

        for (int i = 0; i < edges.size(); i++) {
            final HalfEdge e = edges.get(i);
            final Vertex v0 = e.head();
            final Vertex v1 = e.tail();
            // Find the closest point to each edge, provided it is not closer
            // to an edge adjacent to this one.
            Vertex closest = null;
            double closestDistanceSq = Double.MAX_VALUE;
            for (Vertex v : pointBuffer) {
                if (v != v0 && v != v1) {
                    Vector3d p = Vector3d.nearestPointOnSegment(v0.pnt, v1.pnt, v.pnt);
                    if (p != null) {
                        double d = p.distanceSquared(v.pnt);
                        if (d < closestDistanceSq) {
                            // Check v is not closer to any neighbouring
                            // segments
                            Vertex v2 = e.next.head();
                            Vertex v3 = e.opposite.next.head();
                            assert v2 != v0 && v2 != v1 && v3 != v0 && v3 != v1;
                            p = Vector3d.nearestPointOnSegment(v0.pnt, v2.pnt, v.pnt);
                            if (p != null && p.distanceSquared(v.pnt) < d) {
                                continue;
                            }
                            p = Vector3d.nearestPointOnSegment(v1.pnt, v2.pnt, v.pnt);
                            if (p != null && p.distanceSquared(v.pnt) < d) {
                                continue;
                            }
                            p = Vector3d.nearestPointOnSegment(v0.pnt, v3.pnt, v.pnt);
                            if (p != null && p.distanceSquared(v.pnt) < d) {
                                continue;
                            }
                            p = Vector3d.nearestPointOnSegment(v1.pnt, v3.pnt, v.pnt);
                            if (p != null && p.distanceSquared(v.pnt) < d) {
                                continue;
                            }
                        }
                        closestDistanceSq = d;
                        closest = v;
                    }
                }
            }
            if (closest != null) {
                Vertex v = closest;
                // Here is the threshold test.
                // The original paper would have us test for
                // edge.length() / v.distance(Vector3d.nearestPointToSegment(v0,
                // v1)) < threshold
                // but the angle of deflection is more intuitive and seems more
                // useful too
                double a0 = Vector3d.angle(v.pnt, v0.pnt, v1.pnt);
                double a1 = Vector3d.angle(v.pnt, v1.pnt, v0.pnt);
                if (a1 < threshold && a0 < threshold) {
                    // Split "e" into two. This means removing two faces and
                    // adding four.
                    // Hairy but tested. To understand, sketch it out on paper.
                    // Terminology:
                    // v = breakpoint on edge
                    // v0 = tail of edge
                    // v1 = head of edge
                    // Assuming v0 to v1 is a line going up the page, then...
                    // f0 = face to bottom-left of v0,v1
                    // f1 = face to top-left of v0,v1
                    // f2 = face to bottom-right of v0,v1
                    // f3 = face to top-right of v0,v1
                    // e0 = edge between f0,f1
                    // e1 = edge between f2,f3
                    // e2 = edge between f1,f2
                    // e3 = edge between f2,f3
                    // e4 = edge to top-left of f1
                    // e5 = edge to bottom-left of f0
                    // e6 = edge to bottom-right of f2
                    // e7 = edge to top-right of f3
                    //
                    // Note. assuming normal cartesion with (0,0) and y growing
                    // up, then
                    // edges in faces go CLOCKWISE. Not anti-clockwise, as
                    // stated in Face
                    //
                    Face f0 = Face.createTriangle(v, v0, e.next.vertex);
                    Face f1 = Face.createTriangle(v1, v, e.next.vertex);
                    Face f2 = Face.createTriangle(v0, v, e.opposite.next.vertex);
                    Face f3 = Face.createTriangle(v, v1, e.opposite.next.vertex);
                    // Eight edges to update
                    f0.he0.setOpposite(f1.he0.prev); // e0
                    f2.he0.prev.setOpposite(f3.he0); // e1
                    f1.he0.next.setOpposite(f3.he0.next); // e2
                    f0.he0.next.setOpposite(f2.he0.next); // e3
                    f1.he0.setOpposite(e.prev.opposite); // e4
                    f0.he0.prev.setOpposite(e.next.opposite); // e5
                    f2.he0.setOpposite(e.opposite.prev.opposite); // e6
                    f3.he0.prev.setOpposite(e.opposite.next.opposite); // e7
                    faces.remove(e.face);
                    faces.remove(e.opposite.face);
                    faces.add(f0);
                    faces.add(f1);
                    faces.add(f2);
                    faces.add(f3);
                }
            }
        }
        reindexFacesAndVertices();
    }

    // --------------------------- LOGGING ----------------------
    // Insert your framework of choice logging code here

    private boolean isDebugEnabled() {
        return false;
    }

    private void debug(String s) {
        if (isDebugEnabled()) {
            System.err.println("# " + s);
        }
    }

    private void error(String s) {
        System.err.println("ERROR: " + s);
    }

    // --------------------------- LOGGING ----------------------
}
