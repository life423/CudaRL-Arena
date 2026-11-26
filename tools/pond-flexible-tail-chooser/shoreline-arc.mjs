// Angular extent of one uninterrupted visit to the 0–20 mm clearance band.
export function shorelineArcDegrees(points) {
    let previous = null, angle = 0, min = 0, max = 0, span = 0;
    for (const r of points) {
        if (+r.clearance_bound_mm<=0 || +r.clearance_bound_mm>20) {
            previous=null;
            continue;
        }
        const bearing=Math.atan2(+r.z_m,+r.x_m);
        if (previous===null) angle=min=max=0;
        else angle+=Math.atan2(Math.sin(bearing-previous), Math.cos(bearing-previous));
        min=Math.min(min,angle);
        max=Math.max(max,angle);
        span=Math.max(span,max-min);
        previous=bearing;
    }
    return span*180/Math.PI;
}
